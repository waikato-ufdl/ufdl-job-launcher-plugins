import csv
from glob import glob
import json
import os
import shlex
import traceback
from ufdl.joblauncher import AbstractDockerJobExecutor
from ufdl.pythonclient.functional.object_detection.dataset import download as dataset_download
from ufdl.pythonclient.functional.object_detection.dataset import get_metadata, set_metadata, set_annotations_for_image
from ufdl.pythonclient.functional.core.jobs.job import get_output
from ufdl.json.object_detection import Annotation


class ObjectDetectionTrain_MMDet_20200301(AbstractDockerJobExecutor):
    """
    For executing Tensorflow image classification jobs.
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
        self._log_msg("Downloading dataset:", pk, "-> options=", options, "->", data)
        with open(data, "wb") as zip_file:
            for b in dataset_download(self.context, pk, annotations_args=shlex.split(options)):
                zip_file.write(b)

        # decompress dataset
        output_dir = self.job_dir + "/data"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # replace parameters in template and save it to disk
        template_code = self._expand_template(job, template, bool_to_python=True)
        template_file = self.job_dir + "/output/config.py"
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self._log_file("Template code:", template_file)
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
            self._log_msg("Failed to locate 'labels.txt' file?")
            labels = ["/data/train/labels.txt"]
        else:
            self._log_file("Labels:", labels[0])
            self._log_msg("Using labels from %s" % labels[0])
            labels[0] = labels[0][len(self.job_dir):]
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
            pk, "mmdetlogtxt", "txt",
            glob(self.job_dir + "/output/*.log"),
            self.job_dir + "/log_txt.zip")

        super()._post_run(template, job, pre_run_success, do_run_success, error)


class ObjectDetectionPredict_MMDet_20200301(AbstractDockerJobExecutor):
    """
    For executing Tensorflow image classification prediction jobs.
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

        # download dataset
        data = self.job_dir + "/data.zip"
        pk = int(self._input("data", job, template)["value"])
        options = self._input("data", job, template)["options"]
        self._log_msg("Downloading dataset:", pk, "-> options='" + str(options) + "'", "->", data)
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
            for b in get_output(self.context, pk, "model"):
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

        # build model
        self._run_image(
            image,
            docker_args=docker_args,
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
                    # load CSV file and create annotations
                    annotations = []
                    scores = []
                    csv_file = os.path.splitext(f)[0] + "-rois.csv"
                    with open(csv_file, "r") as cf:
                        reader = csv.DictReader(cf)
                        for row in reader:
                            if ('x' in row) and ('y' in row) and ('w' in row) and ('h' in row) and ('label_str' in row) and ('score' in row):
                                annotation = Annotation(
                                    x=int(float(row['x'])),
                                    y=int(float(row['y'])),
                                    width=int(float(row['w'])),
                                    height=int(float(row['h'])),
                                    label=row['label_str'])
                                if ('poly_x' in row) and ('poly_y' in row):
                                    # TODO
                                    pass
                                annotations.append(annotation)
                                scores.append(float(row['score']))

                    # set annotations for image
                    try:
                        set_annotations_for_image(self.context, dataset_pk, img_name, annotations)
                    except:
                        self._log_msg("Failed to add annotations generated by job %d to dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

                    # set score in metadata
                    try:
                        metadata = get_metadata(self.context, dataset_pk, img_name)
                        if metadata == "":
                            metadata = dict()
                        else:
                            metadata = json.loads(metadata)
                        metadata['scores'] = scores
                        set_metadata(self.context, dataset_pk, img_name, json.dumps(metadata))
                    except:
                        self._log_msg("Failed to add scores of job %d for image %s in dataset %d!\n%s" % (job_pk, img_name, dataset_pk, traceback.format_exc()))

            except:
                self._log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(template, job, pre_run_success, do_run_success, error)
