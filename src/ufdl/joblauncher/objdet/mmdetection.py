from glob import glob
import os
import shlex
import traceback
from typing import Optional, Tuple

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import Parameter, ExtraOutput

from ufdl.jobtypes.base import String, Boolean
from ufdl.jobtypes.standard import Name, PK
from ufdl.jobtypes.standard.container import Array
from ufdl.jobtypes.standard.server import Domain, Framework, PretrainedModel, PretrainedModelInstance
from ufdl.jobtypes.standard.util import BLOB, Nothing

from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download

from wai.json.object import Absent

from ..utils import write_to_file
from .core import read_rois, calculate_confidence_scores, store_annotations, store_scores


DOMAIN_TYPE = Domain("Object Detection")
FRAMEWORK_TYPE = Framework("mmdetection", "2020-03-01")
OBJECT_DETECTION_MMDET_20200301_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class ObjectDetectionTrain_MMDet_20200301(AbstractTrainJobExecutor):
    """
    For executing MMDetection object detection training jobs.
    """
    _cls_contract = Train(OBJECT_DETECTION_MMDET_20200301_CONTRACT_TYPES)

    shared_memory_size: str = Parameter(String())
    pretrained_model: Optional[PretrainedModelInstance] = Parameter(
        PK(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        Name(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE),
        Nothing(),
        default=None
    )

    mmdetlogjson = ExtraOutput(BLOB("json"))
    mmdetlogtxt = ExtraOutput(BLOB("txt"))

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

        # download dataset
        pk: int = self[self.contract.dataset].pk
        data = self._download_dataset(pk)

        # decompress dataset
        output_dir = self.job_dir + "/data"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # download pretrained model
        pretrained = self.pretrained_model
        if pretrained is not None and pretrained.pk is not Absent and pretrained.data:
            pretrained_model = self.job_dir + "/data/pretrained_model.pth"
            self.log_msg("Downloading pretrained model:", pk, "->", pretrained_model)
            with open(pretrained_model, "wb") as pmf:
                for b in pretrainedmodel_download(self.context, pretrained.pk):
                    pmf.write(b)

        # replace parameters in template and save it to disk
        template_code = self._expand_template()
        template_file = self.job_dir + "/output/config.py"
        with open(template_file, "w") as tf:
            tf.write(template_code if isinstance(template_code, str) else "\n".join(template_code))
        self.log_file("Template code:", template_file)
        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """

        image = self.docker_image['url']
        volumes = [
            self.job_dir + "/data" + ":/data",
            self.job_dir + "/output" + ":/output",
        ]
        if self.use_current_user:
            volumes.append(self.cache_dir + ":/.cache")
        else:
            volumes.append(self.cache_dir + ":/root/.cache")

        # build model
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) == 0:
            self.log_msg("Failed to locate 'labels.txt' file?")
            labels = ["/data/train/labels.txt"]
        else:
            self.log_file("Labels:", labels[0])
            self.log_msg("Using labels from %s" % labels[0])
            labels[0] = labels[0][len(self.job_dir):]
        self._fail_on_error(
            self._run_image(
                image,
                docker_args=[
                    "-e", "MMDET_CLASSES=%s" % labels[0],
                    "-e", "MMDET_OUTPUT=/output/",
                    "-e", "MMDET_SETUP=/output/config.py",
                    "-e", "MMDET_DATA=/data"
                ],
                volumes=volumes,
                image_args=[
                    "mmdet_train",
                    "/output/config.py",
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
        # zip+upload model (output_graph.pb and output_labels.txt)
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) == 0:
            labels = [self.job_dir + "/data/train/labels.txt"]
        self._compress_and_upload(
            self.contract.model,
            [
                self.job_dir + "/output/latest.pth",
                self.job_dir + "/output/config.py",
                labels[0]
            ],
            self.job_dir + "/model.zip")

        # zip+upload training logs
        self._compress_and_upload(
            self.mmdetlogjson,
            glob(self.job_dir + "/output/*.log.json"),
            self.job_dir + "/log_json.zip")
        self._compress_and_upload(
            self.mmdetlogtxt,
            glob(self.job_dir + "/output/*.log"),
            self.job_dir + "/log_txt.zip")

        super()._post_run(pre_run_success, do_run_success, error)


class ObjectDetectionPredict_MMDet_20200301(AbstractPredictJobExecutor):
    """
    For executing MMDetection object detection prediction jobs.
    """
    _cls_contract = Predict(OBJECT_DETECTION_MMDET_20200301_CONTRACT_TYPES)

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
        data = self._download_dataset(pk, self.clear_dataset)

        # decompress dataset
        output_dir = self.job_dir + "/prediction/in"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # download model
        model = self.job_dir + "/model.zip"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        # decompress model
        output_dir = self.job_dir + "/output"
        msg = self._decompress(model, output_dir)
        if msg is not None:
            raise Exception("Failed to extract model pk=%d!\n%s" % (pk, msg))

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """

        image = self.docker_image['url']
        volumes = [
            self.job_dir + "/prediction" + ":/prediction",
            self.job_dir + "/output" + ":/output",
        ]
        docker_args = [
            "-e", "MMDET_CLASSES=/output/labels.txt",
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
