from glob import glob
import os
import shlex
import traceback
from typing import Tuple

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import Parameter, ExtraOutput
from ufdl.joblauncher.core.executors.parsers import CommandProgressParser

from ufdl.jobtypes.base import Integer, Boolean, String
from ufdl.jobtypes.standard.container import Array
from ufdl.jobtypes.standard.server import Domain, Framework
from ufdl.jobtypes.standard.util import BLOB

from ufdl.pythonclient.functional.image_classification.dataset import add_categories

from ..utils import write_to_file
from .core import calculate_confidence_scores, read_scores


DOMAIN_TYPE = Domain("Image Classification")
FRAMEWORK_TYPE = Framework("tensorflow", "1.14")
IMAGE_CLASSIFICATION_TF_1_14_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class ImageClassificationTrain_TF_1_14(AbstractTrainJobExecutor):
    """
    For executing Tensorflow image classification training jobs.
    """
    _cls_contract = Train(IMAGE_CLASSIFICATION_TF_1_14_CONTRACT_TYPES)

    steps: int = Parameter(Integer())
    generate_stats: bool = Parameter(Boolean())

    modeltflite = ExtraOutput(BLOB("tficmodeltflite"))
    checkpoint = ExtraOutput(BLOB("tficcheckpoint"))
    log_train = ExtraOutput(BLOB("tensorboard"))
    log_validation = ExtraOutput(BLOB("tensorboard"))
    image_lists = ExtraOutput(BLOB("json"))
    statistics = ExtraOutput(BLOB("csv"))

    def create_command_progress_parser(self) -> CommandProgressParser:
        steps = self.steps
        search_str = ": Step "

        def parser(cmd_output: str, last_progress: float) -> float:
            if search_str in cmd_output:
                step = cmd_output[cmd_output.index(search_str) + len(search_str):]
                if ":" in step:
                    step = step[:step.index(":")]
                try:
                    step = int(step)
                    progress = step / steps * 0.7 + 0.2  # training the only represents 0.7 in the overall train job and starts 0.2
                    if progress != last_progress:
                        self.progress(progress)
                        return progress
                except:
                    pass

            return last_progress

        return parser

    def _pre_run(self):
        """
        Hook method before the actual job is run.

        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run():
            return False

        # download dataset
        pk: int = self[self.contract.dataset].pk
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # create remaining directories
        self._mkdir(self.job_dir + "/output")
        self._mkdir(self.job_dir + "/models")
        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        self.progress(0.1, comment="Getting Docker image...")

        image = self.docker_image['url']
        volumes = [
            self.job_dir + "/data" + ":/data",
            self.job_dir + "/output" + ":/output",
            self.cache_dir + ":/models",
        ]

        self.progress(0.2, comment="Running Docker image...")

        # build model
        image_args = self._expand_template()
        res = self._run_image(
            image,
            volumes=volumes,
            image_args=shlex.split(image_args) if isinstance(image_args, str) else list(image_args),
            command_progress_parser=self.create_command_progress_parser(),
        )

        # export tflite model
        if not self.is_job_cancelled():
            if res is None:
                self.progress(0.9, comment="Exporting tflite model...")
                res = self._run_image(
                    image,
                    volumes=volumes,
                    image_args=[
                        "tfic-export",
                        "--saved_model_dir",
                        "/output/saved_model",
                        "--tflite_model",
                        "/output/saved_model/model.tflite",
                    ],
                )

        # stats?
        if not self.is_job_cancelled():
            if (res is None) and self.generate_stats:
                self.progress(0.95, comment="Generating stats...")
                for t in ["training", "testing", "validation"]:
                    self._run_image(
                        image,
                        volumes=volumes,
                        image_args=[
                            "tfic-stats",
                            "--image_dir", "/data",
                            "--image_list", "/output/%s.json" % t,
                            "--graph_type", "tflite",
                            "--graph", "/output/saved_model/model.tflite",
                            "--info", "/output/info.json",
                            "--output_preds", "/output/%s-predictions.csv" % t,
                            "--output_stats", "/output/%s-stats.csv" % t,
                        ]
                    )

        self.progress(1.0, comment="Done")

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
        if do_run_success:
            self._compress_and_upload(
                self.contract.model,
                [
                    self.job_dir + "/output/graph.pb",
                    self.job_dir + "/output/labels.txt",
                    self.job_dir + "/output/info.json"
                ],
                self.job_dir + "/model.zip"
            )

        # upload tflite model
        if do_run_success:
            self._compress_and_upload(
                self.modeltflite,
                [
                    self.job_dir + "/output/saved_model/model.tflite",
                    self.job_dir + "/output/labels.txt",
                    self.job_dir + "/output/info.json"
                ],
                self.job_dir + "/modeltflite.zip")

        # zip+upload checkpoint (retrain_checkpoint.*)
        if do_run_success:
            self._compress_and_upload(
                self.checkpoint,
                glob(self.job_dir + "/output/retrain_checkpoint.*"),
                self.job_dir + "/checkpoint.zip")

        # zip+upload train/val tensorboard (retrain_logs)
        self._compress_and_upload(
            self.log_train,
            glob(self.job_dir + "/output/retrain_logs/train/events*"),
            self.job_dir + "/tensorboard_train.zip")
        self._compress_and_upload(
            self.log_validation,
            glob(self.job_dir + "/output/retrain_logs/validation/events*"),
            self.job_dir + "/tensorboard_validation.zip")

        # zip+upload train/test/val image list files (*.json)
        self._compress_and_upload(
            self.image_lists,
            glob(self.job_dir + "/output/*.json"),
            self.job_dir + "/image_lists.zip")

        # zip+upload predictions/stats
        if do_run_success and self.generate_stats:
            self._compress_and_upload(
                self.statistics,
                glob(self.job_dir + "/output/*.csv"),
                self.job_dir + "/statistics.zip")

        super()._post_run(pre_run_success, do_run_success, error)


class ImageClassificationPredict_TF_1_14(AbstractPredictJobExecutor):
    """
    For executing Tensorflow image classification prediction jobs.
    """
    _cls_contract = Predict(IMAGE_CLASSIFICATION_TF_1_14_CONTRACT_TYPES)

    store_predictions: bool = Parameter(Boolean())
    confidence_scores: Tuple[str] = Parameter(Array(String()))
    top_x: int = Parameter(Integer())

    predictions = ExtraOutput(BLOB("csv"))

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
        self._mkdir(self.job_dir + "/models")

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

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """

        image: str = self.docker_image['url']
        volumes = [
            self.job_dir + "/prediction" + ":/prediction",
            self.job_dir + "/output" + ":/output",
        ]

        # build model
        image_args = self._expand_template()
        self._run_image(
            image,
            volumes=volumes,
            image_args=shlex.split(image_args) if isinstance(image_args, str) else list(image_args)
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
                self.predictions,
                glob(self.job_dir + "/prediction/out/*.csv"),
                self.job_dir + "/predictions.zip")

        # post-process predictions
        if do_run_success and self.store_predictions:
            try:
                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".csv"):
                        continue
                    img_name = os.path.basename(f)

                    # load CSV file and determine label with highest probability
                    csv_file = os.path.splitext(f)[0] + ".csv"
                    label_scores, label = read_scores(csv_file)

                    # set category for file
                    try:
                        add_categories(self.context, dataset_pk, [img_name], [label])
                    except:
                        self.log_msg("Failed to add labels generated by job %d to dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

                    # calculate confidence scores
                    calculate_confidence_scores(self, dataset_pk, img_name, self.confidence_scores, label_scores)
            except:
                self.log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(pre_run_success, do_run_success, error)
