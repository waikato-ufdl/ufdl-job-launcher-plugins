from glob import glob
import os
import shlex
import traceback
from typing import Tuple

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import ExtraOutput, Parameter
from ufdl.jobtypes.base import Boolean, Integer, String
from ufdl.jobtypes.standard import Name, PK
from ufdl.jobtypes.standard.container import Array

from ufdl.jobtypes.standard.server import Domain, Framework, PretrainedModel, PretrainedModelInstance
from ufdl.jobtypes.standard.util import BLOB

from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download

from ..utils import write_to_file
from .core import (
    AbstractObjDetPredictJobExecutor,
    read_rois,
    calculate_confidence_scores,
    store_annotations,
    store_scores
)


DOMAIN_TYPE = Domain("Object Detection")
FRAMEWORK_TYPE = Framework("yolact", "2020-02-11")
OBJECT_DETECTION_YOLACTPP_20200211_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class ObjectDetectionTrain_YOLACTPP_20200211(AbstractTrainJobExecutor):
    """
    For executing YOLACT++ object detection jobs.
    """
    _cls_contract = Train(OBJECT_DETECTION_YOLACTPP_20200211_CONTRACT_TYPES)

    shared_memory_size: str = Parameter(
        String(),
        default="8G"
    )

    pretrained_model: PretrainedModelInstance = Parameter(
        PK(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        Name(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE))
    )

    validation_epoch: int = Parameter(
        Integer()
    )

    batch_size: int = Parameter(
        Integer()
    )

    yolactpplog = ExtraOutput(BLOB("log"))

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # shared memory size
        self._additional_gpu_flags.extend(["--shm-size", self.shared_memory_size])

        # create directories
        self._mkdir(self.job_dir + "/output")
        self._mkdir(self.job_dir + "/weights")

        # download dataset
        pk: int = self[self.contract.dataset].pk
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # determine labels
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) == 0:
            raise Exception("Failed to locate 'labels.txt' file!")
        self.log_file("Labels:", labels[0])
        with open(labels[0], "r") as lf:
            labels_str = lf.readline()

        # download pretrained model and put it into weights dir
        model_name = 'resnet50-19c8e357'
        model_file = self.job_dir + "/weights/%s.pth" % model_name
        with open(model_file, "wb") as mf:
            for b in pretrainedmodel_download(self.context, self.pretrained_model.pk):
                mf.write(b)

        # replace parameters in template and save it to disk
        template_code = self._expand_template()
        template_code = template_code.replace("${labels}", "'" + "','".join(labels_str.split(",")) + "'")
        template_code = template_code.replace("${num-labels}", str(len(labels_str.split(","))+1))
        template_file = self.job_dir + "/output/config.py"
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self.log_file("Template code:", template_file)
        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """

        image = self.docker_image['url']
        volumes = [
            self.job_dir + "/data" + ":/data",
            self.job_dir + "/weights" + ":/weights",
            self.job_dir + "/output" + ":/output",
        ]
        if self.use_current_user:
            volumes.append(self.cache_dir + ":/.cache")
        else:
            volumes.append(self.cache_dir + ":/root/.cache")

        # build model
        self._fail_on_error(
            self._run_image(
                image,
                docker_args=[
                    "-e", "YOLACTPP_CONFIG=/output/config.py",
                ],
                volumes=volumes,
                image_args=[
                    f"yolactpp_train",
                    f"--config=external_config",
                    f"--log_folder=/output/",
                    f"--save_folder=/weights/",
                    f"--validation_epoch={self.validation_epoch}",
                    f"--batch_size={self.batch_size}",
                ]
            )
        )

    def _post_run(self, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """
        # zip+upload model (latest model, config.py and labels.txt)
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) == 0:
            labels = [self.job_dir + "/data/train/labels.txt"]
        models = glob(self.job_dir + "/weights/model*.pth")
        latest = None
        for model in models:
            if (latest is None) or (os.path.getctime(model) > os.path.getctime(latest)):
                latest = model
        if latest is None:
            latest = self.job_dir + "/output/no_model_found.pth"
        self._compress_and_upload(
            self.contract.model,
            [
                latest,
                self.job_dir + "/output/config.py",
                labels[0]
            ],
            self.job_dir + "/model.zip")

        # zip+upload training logs
        self._compress_and_upload(
            self.yolactpplog,
            glob(self.job_dir + "/output/model.log"),
            self.job_dir + "/model_log.zip")

        super()._post_run(pre_run_success, do_run_success, error)


class ObjectDetectionPredict_YOLACTPP_20200211(AbstractObjDetPredictJobExecutor):
    """
    For executing YOLACT++ object detection prediction jobs.
    """
    _cls_contract = Predict(OBJECT_DETECTION_YOLACTPP_20200211_CONTRACT_TYPES)

    generate_mask_images: bool = Parameter(Boolean())
    store_predictions: bool = Parameter(Boolean())
    confidence_scores: Tuple[str, ...] = Parameter(Array(String()))

    predictions_csv = ExtraOutput(BLOB("csv"))
    predictions_png = ExtraOutput(BLOB("png"))

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # create directories
        self._mkdir(self.job_dir + "/prediction")
        self._mkdir(self.job_dir + "/prediction/in")
        self._mkdir(self.job_dir + "/prediction/out")

        # dataset ID
        pk: int = self[self.contract.dataset].pk

        # download dataset
        output_dir = self.job_dir + "/prediction/in"
        self._download_dataset(pk, output_dir)

        # download model
        model = self.job_dir + "/model.zip"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        # decompress model
        output_dir = self.job_dir + "/output"
        msg = self._decompress(model, output_dir)
        if msg is not None:
            raise Exception("Failed to extract model pk=%d!\n%s" % (pk, msg))
        models = glob(output_dir + "/*.pth")
        if len(models) > 0:
            os.rename(models[0], output_dir + "/latest.pth")

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """

        image = self._docker_image['url']
        volumes = [
            self.job_dir + "/prediction" + ":/prediction",
            self.job_dir + "/output" + ":/output",
        ]
        docker_args = [
            "-e", "YOLACTPP_CONFIG=/output/config.py",
        ]

        # assemble commandline
        cmdline = self._expand_template()

        # use model
        self._fail_on_error(
            self._run_image(
                image,
                docker_args=docker_args,
                volumes=volumes,
                image_args=shlex.split(cmdline) if isinstance(cmdline, str) else cmdline
            )
        )

    def _post_run(self, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """

        dataset_pk: int = self[self.contract.dataset].pk

        # zip+upload predictions
        if do_run_success:
            self._compress_and_upload(
                self.predictions_csv,
                glob(self.job_dir + "/prediction/out/*.csv"),
                self.job_dir + "/predictions.zip")
            if self.generate_mask_images:
                self._compress_and_upload(
                    self.predictions_png,
                    glob(self.job_dir + "/prediction/out/*-mask.png"),
                    self.job_dir + "/predictions_masks.zip")

        # post-process predictions
        if do_run_success and self.store_predictions:
            try:
                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".csv") or f.endswith("-mask.png"):
                        continue
                    img_name = os.path.basename(f)
                    # load CSV file and create annotations
                    csv_file = os.path.splitext(f)[0] + "-rois.csv"
                    annotations, scores = read_rois(csv_file)
                    # set annotations for image
                    store_annotations(self, dataset_pk, img_name, annotations)
                    # set score in metadata
                    store_scores(self, dataset_pk, img_name, scores)
                    # calculate confidence scores
                    calculate_confidence_scores(
                        self,
                        dataset_pk,
                        img_name,
                        self.confidence_scores,
                        annotations,
                        scores
                    )
            except:
                self.log_msg(
                    f"Failed to post-process predictions generated by job {self.job_pk} for dataset {dataset_pk}!\n"
                    f"{traceback.format_exc()}"
                )

        super()._post_run(pre_run_success, do_run_success, error)
