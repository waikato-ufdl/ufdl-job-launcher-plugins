from glob import glob
import os
import traceback
from typing import Optional, Tuple

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
from .core import read_rois, calculate_confidence_scores, store_annotations, store_scores


DOMAIN_TYPE = Domain("Object Detection")
FRAMEWORK_TYPE = Framework("yolo", "v5")
OBJECT_DETECTION_YOLO_V5_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class ObjectDetectionTrain_Yolo_v5(AbstractTrainJobExecutor):
    """
    For executing Tensorflow object detection jobs.
    """
    _cls_contract = Train(OBJECT_DETECTION_YOLO_V5_CONTRACT_TYPES)

    pretrained_model: PretrainedModelInstance = Parameter(
        PK(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        Name(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE))
    )

    image_size: int = Parameter(
        Integer()
    )

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # create directories
        self._mkdir(self.job_dir + "/output")
        self._mkdir(self.job_dir + "/models")
        self._mkdir(self.job_dir + "/data")

        # dataset ID
        pk: int = self.dataset.pk

        # download dataset
        # decompress dataset
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # determine number of classes
        class_labels = []
        with open(os.path.join(self.job_dir, "data", "labels.csv")) as lf:
            lines = lf.readlines()
        for line in lines[1:]:
            index, label = line.strip().split(",", 1)
            class_labels.append(label)
        self.log_msg(f"{len(class_labels)} labels: {class_labels}")

        # download pretrained model and put it into models dir
        model_file = self.job_dir + "/models/pretrained.pt"
        with open(model_file, "wb") as mf:
            for b in pretrainedmodel_download(self.context, self.pretrained_model.pk):
                mf.write(b)

        # replace parameters in template and save it to disk
        template_code = self._expand_template()
        if not isinstance(template_code, str):
            template_code = "\n".join(template_code)
        template_code = template_code.replace("${num-classes}", str(len(class_labels)))
        template_code = template_code.replace("${classes}", str(class_labels))
        template_file = os.path.join(self.job_dir, "data", "dataset.yaml")
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self.log_file("Template code:", template_file)

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        # build model
        self._fail_on_error(
            self._run_image(
                self.docker_image.url,
                volumes=[
                    self.job_dir + "/data" + ":/data",
                    self.job_dir + "/models" + ":/models",
                    self.job_dir + "/output" + ":/output",
                ],
                image_args=[
                    f"yolov5_train",
                    f"--img={self.image_size}",
                    f"--batch=16",
                    f"--epochs=20",
                    f"--data=/data/dataset.yaml",
                    f"--weights=/models/pretrained.pt",
                    f"--project=/output",
                    f"--name=job-number-{self.job_pk}"
                ]
            )
        )

    def _post_run(
            self,
            pre_run_success: bool,
            do_run_success: bool,
            error: Optional[str]
    ):

        if do_run_success:
            # export model
            cmd = [
                f"yolov5_export",
                f"--weights=/output/job-number-{self.job_pk}/weights/best.pt",
                f"--img-size", str(self.image_size), str(self.image_size),
                f"--include onnx"
            ]
            proc = self._execute(cmd)

            if proc.returncode == 0:
                # zip+upload exported model
                path = self.job_dir + "/output/best.onnx"
                zipfile = self.job_dir + "/model.zip"
                self._compress([path], zipfile, strip_path=path)
                self._upload(self.contract.model, zipfile)
            else:
                self._fail_on_error(proc)

        super()._post_run(pre_run_success, do_run_success, error)


class ObjectDetectionPredict_Yolo_v5(AbstractPredictJobExecutor):
    """
    For executing Tensorflow object detection prediction jobs.
    """
    _cls_contract = Predict(OBJECT_DETECTION_YOLO_V5_CONTRACT_TYPES)

    generate_mask_images: bool = Parameter(Boolean())
    store_predictions: bool = Parameter(Boolean())
    confidence_scores: Tuple[str, ...] = Parameter(Array(String()))

    predictions_csv = ExtraOutput(BLOB("csv"))
    predictions_png = ExtraOutput(BLOB("png"))

    image_size: int = Parameter(
        Integer()
    )

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
        self._mkdir(self.job_dir + "/data")
        self._mkdir(self.job_dir + "/output")

        # dataset ID
        pk: int = self.dataset.pk

        # download dataset
        # decompress dataset
        output_dir = self.job_dir + "/prediction/in"
        self._download_dataset(pk, output_dir)

        # determine number of classes
        class_labels = []
        with open(os.path.join(self.job_dir, "data", "labels.csv")) as lf:
            lines = lf.readlines()
        for line in lines[1:]:
            index, label = line.strip().split(",", 1)
            class_labels.append(label)
        self.log_msg(f"{len(class_labels)} labels: {class_labels}")

        # download model
        model = self.job_dir + "/model.zip"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        # decompress model
        output_dir = self.job_dir + "/output"
        msg = self._decompress(model, output_dir)
        if msg is not None:
            raise Exception("Failed to extract model pk=%d!\n%s" % (pk, msg))

        # replace parameters in template and save it to disk
        template_code = self._expand_template()
        if not isinstance(template_code, str):
            template_code = "\n".join(template_code)
        template_code = template_code.replace("${num-classes}", str(len(class_labels)))
        template_code = template_code.replace("${classes}", str(class_labels))
        template_file = os.path.join(self.job_dir, "data", "dataset.yaml")
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self.log_file("Template code:", template_file)

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        self._fail_on_error(
            self._run_image(
                self.docker_image.url,
                volumes=[
                    f"{self.job_dir}/data:/data",
                    f"{self.job_dir}/models:/models",
                    f"{self.job_dir}/output:/output",
                    f"{self.job_dir}/prediction:/prediction"
                ],
                image_args=[
                    f"yolov5_predict_poll",
                    f"--model=/output/best.onnx",
                    f"--data=/data/dataset.yaml",
                    f"--img={self.image_size}",
                    f"--prediction_in=/prediction/in",
                    f"--prediction_out=/prediction/out"
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
