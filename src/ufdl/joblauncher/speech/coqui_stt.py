import re
import tarfile
from glob import glob
import os
import traceback
from typing import Optional, Tuple

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import ExtraOutput, Parameter
from ufdl.joblauncher.core.executors.parsers import CommandProgressParser
from ufdl.jobtypes.base import Boolean, Integer, String
from ufdl.jobtypes.standard import Name, PK
from ufdl.jobtypes.standard.container import Array

from ufdl.jobtypes.standard.server import Domain, Framework, PretrainedModel, PretrainedModelInstance
from ufdl.jobtypes.standard.util import BLOB

from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download

from wai.json.raw import RawJSONObject

from ..utils import write_to_file
from .core import store_transcription


DOMAIN_TYPE = Domain("Speech")
FRAMEWORK_TYPE = Framework("coqui_stt", "1")
SPEECH_COQUI_STT_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}


class SpeechTrain_Coqui_STT(AbstractTrainJobExecutor):
    """
    For executing Coqui STT speech training jobs.
    """
    _cls_contract = Train(SPEECH_COQUI_STT_CONTRACT_TYPES)

    pretrained_model: PretrainedModelInstance = Parameter(
        PK(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        Name(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE))
    )

    epochs: int = Parameter(
        Integer()
    )

    export_quantize: bool = Parameter(
        Boolean()
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
        self.progress(0.05, comment="Downloading dataset...")
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # download pretrained model and put it into models dir
        self.progress(0.15, comment="Downloading pre-trained model...")
        model_file = self.job_dir + "/models/pretrained.tar.gz"
        with open(model_file, "wb") as mf:
            for b in pretrainedmodel_download(self.context, self.pretrained_model.pk):
                mf.write(b)
        tar = tarfile.open(model_file)
        tar.extractall(path=self.job_dir + "/models")
        tar.close()

        # Generate the alphabet
        self.progress(0.2, comment="Generating alphabet...")
        self._fail_on_error(
            self._run_image(
                self.docker_image.url,
                volumes=[
                    self.job_dir + "/data" + ":/data"
                ],
                image_args=[
                    "stt_alphabet",
                    "-i", "/data/train/samples.csv", "/data/val/samples.csv", "/data/test/samples.csv",
                    "-o", "/data/alphabet.txt"
                ]
            )
        )

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        self.progress(0.2, comment="Running Docker image...")

        # build model
        self._fail_on_error(
            self._run_image(
                self.docker_image.url,
                docker_args=["--shm-size", "8G"],
                volumes=[
                    self.job_dir + "/data" + ":/data",
                    self.job_dir + "/models" + ":/models",
                    self.job_dir + "/output" + ":/output",
                ],
                image_args=list(self._expand_template()),
                # TODO: Calculate number of steps from size of dataset and batch size
                command_progress_parser=CoquiSTTTrainCommandProgressParser(self.epochs, 1)
            )
        )

    def _post_run(
            self,
            pre_run_success: bool,
            do_run_success: bool,
            error: Optional[str]
    ):
        if do_run_success:
            self.progress(0.9, comment="Exporting model as TFLite...")

            # export model
            self._fail_on_error(
                self._run_image(
                    self.docker_image.url,
                    docker_args=["--shm-size", "8G"],
                    volumes=[
                        self.job_dir + "/output" + ":/output",
                    ],
                    image_args=[
                        f"stt_export",
                        "--export_quantize", str(self.export_quantize).lower(),
                        "--checkpoint_dir", "/output/model",
                        "--export_dir", "/output/export"
                    ]
                )
            )

            # Upload TFLite model
            self.progress(0.95, comment="Uploading model...")
            self._upload(self.contract.model, f"{self.job_dir}/output/export/output_graph.tflite")

        super()._post_run(pre_run_success, do_run_success, error)


class SpeechPredict_Coqui_STT(AbstractPredictJobExecutor):
    """
    For executing Coqui STT speech prediction jobs.
    """
    _cls_contract = Predict(SPEECH_COQUI_STT_CONTRACT_TYPES)

    store_predictions: bool = Parameter(Boolean())
    confidence_scores: Tuple[str, ...] = Parameter(Array(String()))

    predictions_txt = ExtraOutput(BLOB("txt"))

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

        # download model
        model = self.job_dir + "/output/model.tflite"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        self._fail_on_error(
            self._run_image(
                self.docker_image.url,
                docker_args=["--shm-size", "8G"],
                volumes=[
                    f"{self.job_dir}/data:/data",
                    f"{self.job_dir}/models:/models",
                    f"{self.job_dir}/output:/output",
                    f"{self.job_dir}/prediction:/prediction"
                ],
                image_args=list(self._expand_template())
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
        dataset_pk: int = self.dataset.pk

        # zip+upload predictions
        if do_run_success:
            self._compress_and_upload(
                self.predictions_txt,
                glob(self.job_dir + "/prediction/out/*.txt"),
                self.job_dir + "/predictions.zip")

        # post-process predictions
        if do_run_success and self.store_predictions:
            try:
                for f in glob(self.job_dir + "/prediction/out/*.wav"):
                    wav_name = os.path.basename(f)

                    # load TXT file and create annotations
                    txt_file = os.path.splitext(f)[0] + ".txt"

                    try:
                        with open(txt_file, "r") as file:
                            transcription = file.read()
                    except FileNotFoundError:
                        self.log_msg(f"No matching transcription file found for '{wav_name}'")
                    else:
                        # set annotations for image
                        store_transcription(self, dataset_pk, wav_name, transcription)
            except:
                self.log_msg(
                    f"Failed to post-process predictions generated by job {self.job_pk} for dataset {dataset_pk}!\n"
                    f"{traceback.format_exc()}"
                )

        super()._post_run(pre_run_success, do_run_success, error)


# Regex for parsing the progress command output
COQUI_STT_TRAIN_COMMAND_OUTPUT_REGEX = re.compile(
    r"""
    ^                           # Regex start
    Epoch                       # Epoch keyword
    \s                          # Space
    (?P<epoch_current>\d+)      # The current epoch (0-based)
    .*?                         # Some intervening characters
    Training                    # Training keyword
    .*?                         # Some intervening characters
    Steps:                      # Steps keyword (followed by a colon)
    \s                          # Space
    (?P<step_current>\d+)      # The current step in the current epoch (0-based)
    .*                          # Some trailing characters
    $                           # Regex end
    """,
    flags=re.VERBOSE
)


class CoquiSTTTrainCommandProgressParser(CommandProgressParser):
    def __init__(self, num_epochs: int, num_steps: int):
        self._num_epochs = num_epochs
        self._num_steps = num_steps

    def parse(self, cmd_output: str, last_progress: float) -> Tuple[float, Optional[RawJSONObject]]:
        # See if the output matches the known progress format
        match = COQUI_STT_TRAIN_COMMAND_OUTPUT_REGEX.match(cmd_output)

        # If not, just return the last progress
        if match is None:
            return last_progress, None

        # How many epochs have been completed? (epoch_current is 0-based)
        completed_epochs = int(match.group('epoch_current'))

        # What percentage of epochs have been completed?
        epoch_progress = completed_epochs / self._num_epochs

        # How many steps have been completed? (step_current is 0-based)
        completed_steps = int(match.group('step_current'))

        # If we under-estimated how many steps-per-epoch there are, adjust accordingly
        if completed_steps > self._num_steps:
            self._num_steps = completed_steps

        # What percentage of the way through the current epoch are we?
        step_progress = completed_steps / self._num_steps

        # What percentage of the way through the overall training process are we?
        overall_progress = epoch_progress + step_progress / self._num_epochs

        # Train command represents only 70% of the overall job execution
        return overall_progress * 0.7 + 0.2, None
