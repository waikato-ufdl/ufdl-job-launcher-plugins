import csv
from glob import glob
import os
import shlex
import traceback
from ufdl.joblauncher.core import AbstractDockerJobExecutor, Input, JobOutput
from ufdl.pythonclient.functional.image_classification.dataset import download as dataset_download
from ufdl.pythonclient.functional.image_classification.dataset import clear as dataset_clear
from ufdl.pythonclient.functional.image_classification.dataset import add_categories
from .core import calculate_confidence_scores, read_scores


def imageclassifiation_tf_1_14_command_progress_parser(job_executor, job, cmd_output, last_progress):
    """
    Provides updates on the training process.

    :param job_executor: the reference to the job executor calling this method
    :type job_executor: AbstractJobExecutor
    :param job: the current job
    :type job: dict
    :param cmd_output: the command output string to process
    :type cmd_output: str
    :param last_progress: the last reported progress (0-1)
    :type last_progress: float
    :return: returns the progress (0-1)
    :rtype: float
    """
    steps = int(job['parameter_values']["steps"])
    job_pk = int(job['pk'])
    search_str = ": Step "
    if search_str in cmd_output:
        step = cmd_output[cmd_output.index(search_str) + len(search_str):]
        if ":" in step:
            step = step[:step.index(":")]
        try:
            step = int(step)
            progress = step / steps * 0.7 + 0.2  # training the only represents 0.7 in the overall train job and starts 0.2
            if progress != last_progress:
                job_executor.progress(job_pk, progress)
                return progress
        except:
            pass
    return last_progress


class ImageClassificationTrain_TF_1_14(AbstractDockerJobExecutor):
    """
    For executing Tensorflow image classification training jobs.
    """

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ImageClassificationTrain_TF_1_14, self).__init__(context, config)

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

        # download dataset
        data = self.job_dir + "/data.zip"
        pk = int(self._input("data", job, template)["value"])
        options = self._input("data", job, template)["options"]
        self.log_msg("Downloading dataset:", pk, "-> options='" + str(options) + "'", "->", data)
        with open(data, "wb") as zip_file:
            for b in dataset_download(self.context, pk, annotations_args=shlex.split(options)):
                zip_file.write(b)

        # decompress dataset
        output_dir = self.job_dir + "/data"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # create remaining directories
        self._mkdir(self.job_dir + "/output")
        self._mkdir(self.job_dir + "/models")
        return True

    def _do_run(self, template, job):
        """
        Executes the actual job. Only gets run if pre-run was successful.

        :param template: the job template to apply
        :type template: dict
        :param job: the job with the actual values for inputs and parameters
        :type job: dict
        """

        job_pk = int(job['pk'])

        self.progress(job_pk, 0.1, comment="Getting Docker image...")

        image = self.docker_image['url']
        volumes = [
            self.job_dir + "/data" + ":/data",
            self.job_dir + "/output" + ":/output",
            self.cache_dir + ":/models",
        ]

        self.progress(job_pk, 0.2, comment="Running Docker image...")

        # build model
        res = self._run_image(
            image,
            volumes=volumes,
            image_args=shlex.split(self._expand_template(job, template)),
            command_progress_parser=imageclassifiation_tf_1_14_command_progress_parser,
            job=job
        )

        # export tflite model
        if not self.is_job_cancelled(job):
            if res is None:
                self.progress(job_pk, 0.9, comment="Exporting tflite model...")
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
        if not self.is_job_cancelled(job):
            if (res is None) and (self._parameter("generate-stats", job, template)['value'] == "true"):
                self.progress(job_pk, 0.95, comment="Generating stats...")
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

        self.progress(job_pk, 1.0, comment="Done")

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

        # zip+upload model (output_graph.pb and output_labels.txt)
        if do_run_success:
            self._compress_and_upload(
                job_pk, "model", "tficmodel",
                [
                    self.job_dir + "/output/graph.pb",
                    self.job_dir + "/output/labels.txt",
                    self.job_dir + "/output/info.json"
                ],
                self.job_dir + "/model.zip")

        # upload tflite model
        if do_run_success:
            self._compress_and_upload(
                job_pk, "modeltflite", "tficmodeltflite",
                [
                    self.job_dir + "/output/saved_model/model.tflite",
                    self.job_dir + "/output/labels.txt",
                    self.job_dir + "/output/info.json"
                ],
                self.job_dir + "/modeltflite.zip")

        # zip+upload checkpoint (retrain_checkpoint.*)
        if do_run_success:
            self._compress_and_upload(
                job_pk, "checkpoint", "tficcheckpoint",
                glob(self.job_dir + "/output/retrain_checkpoint.*"),
                self.job_dir + "/checkpoint.zip")

        # zip+upload train/val tensorboard (retrain_logs)
        self._compress_and_upload(
            job_pk, "log_train", "tensorboard",
            glob(self.job_dir + "/output/retrain_logs/train/events*"),
            self.job_dir + "/tensorboard_train.zip")
        self._compress_and_upload(
            job_pk, "log_validation", "tensorboard",
            glob(self.job_dir + "/output/retrain_logs/validation/events*"),
            self.job_dir + "/tensorboard_validation.zip")

        # zip+upload train/test/val image list files (*.json)
        self._compress_and_upload(
            job_pk, "image_lists", "json",
            glob(self.job_dir + "/output/*.json"),
            self.job_dir + "/image_lists.zip")

        # zip+upload predictions/stats
        if do_run_success and (self._parameter("generate-stats", job, template)['value'] == "true"):
            self._compress_and_upload(
                job_pk, "statistics", "csv",
                glob(self.job_dir + "/output/*.csv"),
                self.job_dir + "/statistics.zip")

        super()._post_run(template, job, pre_run_success, do_run_success, error)


class ImageClassificationPredict_TF_1_14(AbstractDockerJobExecutor):
    """
    For executing Tensorflow image classification prediction jobs.
    """
    # The model to use for prediction (must come from a job output)
    model: JobOutput = Input({
        "job_output<tficmodel>": JobOutput,
        "job_output<tficmodeltflite>": JobOutput
    })

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ImageClassificationPredict_TF_1_14, self).__init__(context, config)

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
        self._mkdir(self.job_dir + "/models")

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
        with open(model, "wb") as zip_file:
            for b in self.model.download():
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

        image = self.docker_image['url']
        volumes=[
            self.job_dir + "/prediction" + ":/prediction",
            self.job_dir + "/output" + ":/output",
        ]

        # build model
        self._run_image(
            image,
            volumes=volumes,
            image_args=shlex.split(self._expand_template(job, template))
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

        # post-process predictions
        if do_run_success and (self._parameter('store-predictions', job, template)['value'] == "true"):
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
                        self.log_msg("Failed to add labels generated by job %d to dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

                    # calculate confidence scores
                    calculate_confidence_scores(self, job_pk, dataset_pk, img_name, self._parameter('confidence-scores', job, template)['value'].split(";"), label_scores)
            except:
                self.log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(template, job, pre_run_success, do_run_success, error)
