import json
from glob import glob
import os
import shlex
import traceback
from typing import List, Optional, Tuple

from ufdl.jobcontracts.standard import Train, Predict
from ufdl.joblauncher.core.config import UFDLJobLauncherConfig

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import Parameter, ExtraOutput
from ufdl.joblauncher.core.executors.parsers import CommandProgressParser
from ufdl.joblauncher.core.types import Job, Template

from ufdl.jobtypes.base import Integer, Boolean, String
from ufdl.jobtypes.standard import Name, PK
from ufdl.jobtypes.standard.server import DockerImage, Domain, Framework, PretrainedModel, PretrainedModelInstance
from ufdl.jobtypes.standard.util import BLOB, Nothing
from ufdl.pythonclient import UFDLServerContext

from ufdl.pythonclient.functional.image_classification.dataset import add_categories

from ..utils import write_to_file
from .core import calculate_confidence_scores, read_scores


DOMAIN_TYPE = Domain("Image Classification")

FRAMEWORK_TYPE_0_23_1 = Framework((String(), String.generate_subclass('mmclass_0.23.1')()))
FRAMEWORK_TYPE_0_25_0 = Framework((String(), String.generate_subclass('mmclass_0.25.0')()))

IMAGE_CLASSIFICATION_MMCLASS_0_23_1_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE_0_23_1}
IMAGE_CLASSIFICATION_MMCLASS_0_25_0_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE_0_25_0}

# The docker-image framework doesn't match the model framework
DOCKER_IMAGE_TYPE_0_23_1 = DockerImage(DOMAIN_TYPE, Framework('mmclass', '0.23.1'))
DOCKER_IMAGE_TYPE_0_25_0 = DockerImage(DOMAIN_TYPE, Framework('mmclass', '0.25.0'))


class AbstractImageClassificationTrainMMClass(AbstractTrainJobExecutor):
    """
    For executing MMClassification image classification training jobs.
    """
    epochs: int = Parameter(Integer())

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job,
            docker_image: DockerImage
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, docker_image)

        # Need to dynamically assign the pretrained model, as its framework type
        # depends on the specifc framework type we are implementing, not the
        # base type
        model_framework = self._extract_docker_image_type_from_contract(self._contract).type_args[1]
        self.pretrained_model: Optional[PretrainedModelInstance] = Parameter(
            PK(PretrainedModel(DOMAIN_TYPE, model_framework)),
            Name(PretrainedModel(DOMAIN_TYPE, model_framework)),
            PretrainedModel(DOMAIN_TYPE, model_framework),
            Nothing(),
            default=None
        )

        self.labels: List[str] = []

    def create_command_progress_parser(self) -> CommandProgressParser:
        """
        Looks for the epoch in the output.
        """
        def parser(cmd_output: str, last_progress: float) -> Tuple[float, None]:
            if " - mmcls - INFO - Epoch(val) [" in cmd_output:
                s = cmd_output
                f = "Epoch(val) ["
                s = s[s.index(f) + len(f):]
                if "]" in s:
                    s = s[0:s.index("]")]
                    try:
                        return float(s) / self.epochs, None
                    except:
                        pass
            return last_progress, None

        return CommandProgressParser.from_callable(parser)

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

        # download dataset
        pk: int = self[self.contract.dataset].pk
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # Read in the labels
        with open(self.job_dir + "/data/labels.txt", "r") as labels_file:
            self.labels = labels_file.read().split(',')

        # Format the config file and write it to disk
        config = self._expand_template({
            "num-classes": len(self.labels),
            "pretrained_model_config": (
                ""
                if self.pretrained_model is None else
                f"init_cfg=dict("
                f"type='Pretrained',"
                f"checkpoint='{self.pretrained_model.url}',"
                f"prefix='backbone'),"
            )
        })
        with open(self.job_dir + "/output/config.py", "w") as config_file:
            if isinstance(config, str):
                config_file.write(config)
            else:
                config_file.writelines(f"{line}\n" for line in config)

        return True

    def _do_run(self):
        """
        Executes the actual job. Only gets run if pre-run was successful.
        """
        self.progress(0.1, comment="Getting Docker image...")

        image = self.docker_image['url']
        volumes = [
            self.job_dir + ":/workspace",
            self.cache_dir + ":/.cache",
        ]

        self.progress(0.2, comment="Running Docker image...")

        # build model
        res = self._run_image(
            image,
            docker_args=["-e", f"MMCLS_CLASSES={','.join(self.labels)}"],
            volumes=volumes,
            image_args=[
                "mmcls_train",
                "/workspace/output/config.py",
                "--work-dir", "/workspace/output"
            ],
            command_progress_parser=self.create_command_progress_parser(),
        )

        if res is not None:
            res.check_returncode()

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
        # zip+upload model
        if do_run_success:
            self._compress_and_upload(
                self.contract.model,
                [
                    self.job_dir + "/output/latest.pth",
                    self.job_dir + "/output/config.py",
                    self.job_dir + "/data/labels.txt"
                ],
                self.job_dir + "/model.zip"
            )

        super()._post_run(pre_run_success, do_run_success, error)


class AbstractImageClassificationPredictMMClass(AbstractPredictJobExecutor):
    """
    For executing MMClassification image classification prediction jobs.
    """

    store_predictions: bool = Parameter(Boolean())

    predictions = ExtraOutput(BLOB("json"))

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job,
            docker_image: DockerImage
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, docker_image)

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
            self.job_dir + ":/workspace"
        ]

        # Read in the list of labels
        with open(self.job_dir + "/output/labels.txt", "r") as labels_file:
            labels = labels_file.read()

        # build model
        image_args = self._expand_template()
        self._run_image(
            image,
            docker_args=["-e", f"MMCLS_CLASSES={labels}"],
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
                glob(self.job_dir + "/prediction/out/*.json"),
                self.job_dir + "/predictions.zip")

        # post-process predictions
        # FIXME: Copy-pasted from ../tensorflow.py, needs updating for MMClassification
        if do_run_success and self.store_predictions:
            try:
                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".json"):
                        continue
                    img_name = os.path.basename(f)

                    # load JSON file and determine label with highest probability
                    json_file = os.path.splitext(f)[0] + ".json"
                    with open(json_file, "r") as file:
                        label_scores = json.load(file)
                    label = max(
                        label_scores.items(),
                        default=(None,),
                        key=lambda t: t[1]
                    )[0]

                    # set category for file
                    if label is not None:
                        try:
                            add_categories(self.context, dataset_pk, [img_name], [label])
                        except:
                            self.log_msg("Failed to add labels generated by job %d to dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

                    # TODO: Reintroduce
                    # calculate confidence scores
                    # calculate_confidence_scores(self, dataset_pk, img_name, self.confidence_scores, label_scores)
            except:
                self.log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (self.job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(pre_run_success, do_run_success, error)


class ImageClassificationTrain_MMClass_0_23_1(AbstractImageClassificationTrainMMClass):
    """
    For executing MMClassification image classification training jobs.
    """
    _cls_contract = Train(IMAGE_CLASSIFICATION_MMCLASS_0_23_1_CONTRACT_TYPES)

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, DOCKER_IMAGE_TYPE_0_23_1)


class ImageClassificationPredict_MMClass_0_23_1(AbstractImageClassificationPredictMMClass):
    """
    For executing MMClassification image classification prediction jobs.
    """
    _cls_contract = Predict(IMAGE_CLASSIFICATION_MMCLASS_0_23_1_CONTRACT_TYPES)

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, DOCKER_IMAGE_TYPE_0_23_1)


class ImageClassificationTrain_MMClass_0_25_0(AbstractImageClassificationTrainMMClass):
    """
    For executing MMClassification image classification training jobs.
    """
    _cls_contract = Train(IMAGE_CLASSIFICATION_MMCLASS_0_25_0_CONTRACT_TYPES)

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, DOCKER_IMAGE_TYPE_0_25_0)


class ImageClassificationPredict_MMClass_0_25_0(AbstractImageClassificationPredictMMClass):
    """
    For executing MMClassification image classification prediction jobs.
    """
    _cls_contract = Predict(IMAGE_CLASSIFICATION_MMCLASS_0_25_0_CONTRACT_TYPES)

    def __init__(
            self,
            context: UFDLServerContext,
            config: UFDLJobLauncherConfig,
            template: Template,
            job: Job
    ):
        # Need to override the default docker-image type,
        # as the default's framework type is the same as
        # the model's framework type, but MMClassification
        # can work with many models.
        super().__init__(context, config, template, job, DOCKER_IMAGE_TYPE_0_25_0)
