from glob import glob
import json
import os
import shlex
import traceback
from ufdl.joblauncher.core import AbstractDockerJobExecutor
from ufdl.pythonclient.functional.object_detection.dataset import download as dataset_download
from ufdl.pythonclient.functional.object_detection.dataset import clear as dataset_clear
from ufdl.pythonclient.functional.core.jobs.job import get_output
from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download
from ufdl.pythonclient.functional.core.models.pretrained_model import retrieve as pretrainedmodel_retrieve
from .core import read_rois, calculate_confidence_scores, store_annotations, store_scores


class ObjectDetectionTrain_MMDet_20200301(AbstractDockerJobExecutor):
    """
    For executing MMDetection object detection training jobs.
    """

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ObjectDetectionTrain_MMDet_20200301, self).__init__(context, config)

    def _pre_run(self, template, job):
        """
        Hook method before the actual job is run.

        :param template: the job template to apply
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run(template, job):
            return False

        # shared memory size
        try:
            shm_size = self._parameter('shared-memory-size', job, template)['value']
        except:
            shm_size = "8G"
        self._additional_gpu_flags.extend(["--shm-size", shm_size])

        # create directories
        self._mkdir(self.job_dir + "/output")

        # download dataset
        data = self.job_dir + "/data.zip"
        pk = int(self._input("data", job, template)["value"])
        options = self._input("data", job, template)["options"]
        self.log_msg("Downloading dataset:", pk, "-> options=", options, "->", data)
        with open(data, "wb") as zip_file:
            for b in dataset_download(self.context, pk, annotations_args=shlex.split(options)):
                zip_file.write(b)

        # decompress dataset
        output_dir = self.job_dir + "/data"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # download pretrained model
        try:
            pretrained = self._parameter('pretrained-model', job, template)['value']
            pretrained = int(pretrained)
        except:
            # not all templates may specify that parameter, so we'll just ignore
            # the "param not found" exception
            pretrained = None

        if pretrained is not None:
            pretrained_model = self.job_dir + "/data/pretrained_model.pth"
            self.log_msg("Downloading pretrained model:", pk, "->", pretrained_model)
            with open(pretrained_model, "wb") as pmf:
                for b in pretrainedmodel_download(self.context, pretrained):
                    pmf.write(b)

        # replace parameters in template and save it to disk
        template_code = self._expand_template(job, template, bool_to_python=True)
        template_file = self.job_dir + "/output/config.py"
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self.log_file("Template code:", template_file)
        return True

    def _do_run(self, template, job):
        """
        Executes the actual job. Only gets run if pre-run was successful.

        :param template: the job template to apply
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        """

        image = self._docker_image['url']
        volumes=[
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

    def _post_run(self, template, job, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param template: the job template that was applied
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """

        pk = int(job['pk'])

        # zip+upload model (output_graph.pb and output_labels.txt)
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) == 0:
            labels = [self.job_dir + "/data/train/labels.txt"]
        self._compress_and_upload(
            pk, "model", "mmdetmodel",
            [
                self.job_dir + "/output/latest.pth",
                self.job_dir + "/output/config.py",
                labels[0]
            ],
            self.job_dir + "/model.zip")

        # zip+upload training logs
        self._compress_and_upload(
            pk, "mmdetlog", "json",
            glob(self.job_dir + "/output/*.log.json"),
            self.job_dir + "/log_json.zip")
        self._compress_and_upload(
            pk, "mmdetlog", "txt",
            glob(self.job_dir + "/output/*.log"),
            self.job_dir + "/log_txt.zip")

        super()._post_run(template, job, pre_run_success, do_run_success, error)


class ObjectDetectionPredict_MMDet_20200301(AbstractDockerJobExecutor):
    """
    For executing MMDetection object detection prediction jobs.
    """

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ObjectDetectionPredict_MMDet_20200301, self).__init__(context, config)

    def _pre_run(self, template, job):
        """
        Hook method before the actual job is run.

        :param template: the job template to apply
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        :return: whether successful
        :rtype: bool
        """
        if not super()._pre_run(template, job):
            return False

        # create directories
        self._mkdir(self.job_dir + "/prediction")
        self._mkdir(self.job_dir + "/prediction/in")
        self._mkdir(self.job_dir + "/prediction/out")

        # dataset ID
        pk = int(self._input("data", job, template)["value"])

        # clear dataset
        if self._parameter('clear-dataset', job, template)['value'] == "true":
            dataset_clear(self.context, pk)

        # download dataset
        data = self.job_dir + "/data.zip"
        options = self._input("data", job, template)["options"]
        self.log_msg("Downloading dataset:", pk, "-> options='" + str(options) + "'", "->", data)
        with open(data, "wb") as zip_file:
            for b in dataset_download(self.context, pk, annotations_args=shlex.split(options)):
                zip_file.write(b)

        # decompress dataset
        output_dir = self.job_dir + "/prediction/in"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # download model
        model = self.job_dir + "/model.zip"
        pk = self._pk_from_joboutput(self._input("model", job, template)["value"])
        with open(model, "wb") as zip_file:
            for b in get_output(self.context, pk, "model", "mmdetmodel"):
                zip_file.write(b)

        # decompress model
        output_dir = self.job_dir + "/output"
        msg = self._decompress(model, output_dir)
        if msg is not None:
            raise Exception("Failed to extract model pk=%d!\n%s" % (pk, msg))

        return True

    def _do_run(self, template, job):
        """
        Executes the actual job. Only gets run if pre-run was successful.

        :param template: the job template to apply
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        """

        image = self._docker_image['url']
        volumes=[
            self.job_dir + "/prediction" + ":/prediction",
            self.job_dir + "/output" + ":/output",
        ]
        docker_args = [
            "-e", "MMDET_CLASSES=/output/labels.txt",
        ]

        # assemble commandline
        cmdline = self._expand_template(job, template)
        if self._is_true('generate-mask-images', job, template):
            cmdline += " --output_mask_image"

        # use model
        self._fail_on_error(
            self._run_image(
                image,
                docker_args=docker_args,
                volumes=volumes,
                image_args=shlex.split(cmdline)
            )
        )


    def _post_run(self, template, job, pre_run_success, do_run_success, error):
        """
        Hook method after the actual job has been run. Will always be executed.

        :param template: the job template that was applied
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        :param pre_run_success: whether the pre_run code was successfully run
        :type pre_run_success: bool
        :param do_run_success: whether the do_run code was successfully run (only gets run if pre-run was successful)
        :type do_run_success: bool
        :param error: any error that may have occurred, None if none occurred
        :type error: str
        """

        job_pk = int(job['pk'])
        dataset_pk = int(self._input("data", job, template)["value"])

        # zip+upload predictions
        if do_run_success:
            self._compress_and_upload(
                job_pk, "predictions", "csv",
                glob(self.job_dir + "/prediction/out/*.csv"),
                self.job_dir + "/predictions.zip")
            if self._is_true('generate-mask-images', job, template):
                self._compress_and_upload(
                    job_pk, "predictions", "png",
                    glob(self.job_dir + "/prediction/out/*-mask.png"),
                    self.job_dir + "/predictions_masks.zip")

        # post-process predictions
        if do_run_success and (self._parameter('store-predictions', job, template)['value'] == "true"):
            try:
                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".csv") or f.endswith("-mask.png"):
                        continue
                    img_name = os.path.basename(f)
                    # load CSV file and create annotations
                    csv_file = os.path.splitext(f)[0] + "-rois.csv"
                    annotations, scores = read_rois(csv_file)
                    # set annotations for image
                    store_annotations(self, job_pk, dataset_pk, img_name, annotations)
                    # set score in metadata
                    store_scores(self, job_pk, dataset_pk, img_name, scores)
                    # calculate confidence scores
                    calculate_confidence_scores(self, job_pk, dataset_pk, img_name, self._parameter('confidence-scores', job, template)['value'].split(";"), annotations, scores)
            except:
                self.log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(template, job, pre_run_success, do_run_success, error)
