import csv
from glob import glob
import json
import os
import shlex
import shutil
import tarfile
import traceback
from ufdl.joblauncher import AbstractDockerJobExecutor, load_class
from ufdl.pythonclient.functional.image_classification.dataset import download as ic_dataset_download
from ufdl.pythonclient.functional.image_classification.dataset import add_categories as ic_add_categories
from ufdl.pythonclient.functional.image_classification.dataset import get_metadata as ic_get_metadata
from ufdl.pythonclient.functional.image_classification.dataset import set_metadata as ic_set_metadata
from ufdl.pythonclient.functional.object_detection.dataset import download as od_dataset_download
from ufdl.pythonclient.functional.object_detection.dataset import get_metadata as od_get_metadata
from ufdl.pythonclient.functional.object_detection.dataset import set_metadata as od_set_metadata
from ufdl.pythonclient.functional.object_detection.dataset import set_annotations_for_image as od_set_annotations_for_image
from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download
from ufdl.pythonclient.functional.core.jobs.job import get_output
from ufdl.json.object_detection import Annotation


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
        self._log_msg("Downloading dataset:", pk, "-> options='" + str(options) + "'", "->", data)
        with open(data, "wb") as zip_file:
            for b in ic_dataset_download(self.context, pk, annotations_args=shlex.split(options)):
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

        image = self._docker_image['url']
        volumes=[
            self.job_dir + "/data" + ":/data",
            self.job_dir + "/output" + ":/output",
            self.cache_dir + ":/models",
        ]

        # build model
        res = self._run_image(
            image,
            volumes=volumes,
            image_args=shlex.split(self._expand_template(job, template))
        )

        # stats?
        if (res is None) and (self._parameter("generate-stats", job, template)['value'] == "true"):
            for t in ["training", "testing", "validation"]:
                self._run_image(
                    image,
                    volumes=volumes,
                    image_args=[
                        "tfic-stats",
                        "--image_dir", "/data",
                        "--image_list", "/output/%s.json" % t,
                        "--graph", "/output/graph.pb",
                        "--info", "/output/info.json",
                        "--output_preds", "/output/%s-predictions.csv" % t,
                        "--output_stats", "/output/%s-stats.csv" % t,
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

        # download dataset
        data = self.job_dir + "/data.zip"
        pk = int(self._input("data", job, template)["value"])
        options = self._input("data", job, template)["options"]
        self._log_msg("Downloading dataset:", pk, "-> options='" + str(options) + "'", "->", data)
        with open(data, "wb") as zip_file:
            for b in ic_dataset_download(self.context, pk, annotations_args=shlex.split(options)):
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
            conf_score_str = self._parameter('confidence-scores', job, template)['value'].split(";")
            conf_score_obj = []
            try:
                for c in conf_score_str:
                    conf_score_obj.append(load_class(c)())
            except:
                self._log_msg("Failed to instantiate confidence score classes: %s\n%s" % (conf_score_str, traceback.format_exc()))

            try:
                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".csv"):
                        continue
                    # load CSV file and determine label with highest probability
                    csv_file = os.path.splitext(f)[0] + ".csv"
                    label = "?"
                    prob = 0.0
                    label_scores = dict()
                    with open(csv_file, "r") as cf:
                        reader = csv.DictReader(cf)
                        for row in reader:
                            if ('probability' in row) and ('label' in row):
                                label_scores[row['label']] = float(row['probability'])
                                if float(row['probability']) > prob:
                                    prob = float(row['probability'])
                                    label = row['label']

                    # set category for file
                    try:
                        img_name = os.path.basename(f)
                        ic_add_categories(self.context, dataset_pk, [img_name], [label])
                    except:
                        self._log_msg("Failed to add labels generated by job %d to dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

                    # calculate confidence scores
                    if len(conf_score_obj) > 0:
                        try:
                            conf_scores = dict()
                            for c in conf_score_obj:
                                conf_scores[c.name()] = c.calculate(label_scores)
                            metadata = ic_get_metadata(self.context, dataset_pk, img_name)
                            if metadata == "":
                                metadata = dict()
                            else:
                                metadata = json.loads(metadata)
                            metadata['confidence'] = conf_scores
                            ic_set_metadata(self.context, dataset_pk, img_name, json.dumps(metadata))
                        except:
                            self._log_msg("Failed to add confidence scores of job %d for image %s in dataset %d!\n%s" % (job_pk, img_name, dataset_pk, traceback.format_exc()))
            except:
                self._log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(template, job, pre_run_success, do_run_success, error)


class ObjectDetectionTrain_TF_1_14(AbstractDockerJobExecutor):
    """
    For executing Tensorflow object detection jobs.
    """

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ObjectDetectionTrain_TF_1_14, self).__init__(context, config)

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
        self._mkdir(self.job_dir + "/output")
        self._mkdir(self.job_dir + "/models")

        # download dataset
        data = self.job_dir + "/data.zip"
        pk = int(self._input("data", job, template)["value"])
        options = self._input("data", job, template)["options"]
        self._log_msg("Downloading dataset:", pk, "-> options=", options, "->", data)
        with open(data, "wb") as zip_file:
            for b in od_dataset_download(self.context, pk, annotations_args=shlex.split(options)):
                zip_file.write(b)

        # decompress dataset
        output_dir = self.job_dir + "/data"
        msg = self._decompress(data, output_dir)
        if msg is not None:
            raise Exception("Failed to extract dataset pk=%d!\n%s" % (pk, msg))

        # determine number of classes
        num_classes = 0
        class_labels = []
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) > 0:
            with open(labels[0]) as lf:
                lines = lf.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("name:"):
                    num_classes += 1
                    class_labels.append(line[5:].strip())
        self._log_msg("%s labels: %s" % (str(num_classes), str(class_labels)))

        # download pretrained model and put it into models dir
        model_file = self.job_dir + "/models/pretrained.tar.gz"
        with open(model_file, "wb") as mf:
            for b in pretrainedmodel_download(self.context, int(self._parameter('pretrained-model', job, template)['value'])):
                mf.write(b)
        tar = tarfile.open(model_file)
        tar.extractall()
        tar.close()
        # rename model dir to "pretrainedmodel"
        path = self.job_dir + "/models"
        for f in os.listdir(path):
            d = os.path.join(path, f)
            if os.path.isdir(d):
                os.rename(d, os.path.join(self.job_dir, "models", "pretrainedmodel"))
                break

        # replace parameters in template and save it to disk
        template_code = self._expand_template(job, template)
        template_code = template_code.replace("${num-classes}", str(num_classes))
        template_file = self.job_dir + "/output/pipeline.config"
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
            self.job_dir + "/models" + ":/models",
            self.job_dir + "/output" + ":/output",
        ]

        # build model
        self._run_image(
            image,
            volumes=volumes,
            image_args=[
                "objdet_train",
                "--pipeline_config_path=/output/pipeline.config",
                "--model_dir=/output/",
                "--num_train_steps=%s" %  self._parameter('num-train-steps', job, template)['value'],
                "--sample_1_of_n_eval_examples=1",
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

        # export model
        cmd = [
            "objdet_export",
            "--input_type image_tensor",
            "--pipeline_config_path /output/pipeline.config",
            "--trained_checkpoint_prefix /output/model.ckpt-%s" %  self._parameter('num-train-steps', job, template)['value'],
            "--output_directory /output/exported_graphs"
        ]
        self._execute(cmd)

        # zip+upload model
        path = self.job_dir + "/output/exported_graphs"
        labels = glob(self.job_dir + "/**/labels.txt", recursive=True)
        if len(labels) > 0:
            shutil.copyfile(labels[0], os.path.join(path, "labels.txt"))
        zipfile = self.job_dir + "/model.zip"
        self._compress(glob(path, recursive=True), zipfile, strip_path=path)
        self._upload(pk, "model", "tfodmodel", zipfile)

        # zip+upload checkpoint
        path = self.job_dir + "/output"
        files = glob(path + "/model.ckpt-%s" % self._parameter('num-train-steps', job, template)['value'])
        files.append(path + "/graph.pbtxt")
        files.append(path + "/pipeline.config")
        self._compress_and_upload(pk, "checkpoint", "tfodcheckpoint", files, self.job_dir + "/checkpoint.zip")

        super()._post_run(template, job, pre_run_success, do_run_success, error)


class ObjectDetectionPredict_TF_1_14(AbstractDockerJobExecutor):
    """
    For executing Tensorflow object detection prediction jobs.
    """

    def __init__(self, context, config):
        """
        Initializes the executor with the backend context.

        :param context: the server context
        :type context: UFDLServerContext
        :param config: the configuration to use
        :type config: configparser.ConfigParser
        """
        super(ObjectDetectionPredict_TF_1_14, self).__init__(context, config)

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
            for b in od_dataset_download(self.context, pk, annotations_args=shlex.split(options)):
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
        # TODO
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
            "-e", "YOLACTPP_CONFIG=/output/config.py",
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
                        od_set_annotations_for_image(self.context, dataset_pk, img_name, annotations)
                    except:
                        self._log_msg("Failed to add annotations generated by job %d to dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

                    # set score in metadata
                    try:
                        metadata = od_get_metadata(self.context, dataset_pk, img_name)
                        if metadata == "":
                            metadata = dict()
                        else:
                            metadata = json.loads(metadata)
                        metadata['scores'] = scores
                        od_set_metadata(self.context, dataset_pk, img_name, json.dumps(metadata))
                    except:
                        self._log_msg("Failed to add scores of job %d for image %s in dataset %d!\n%s" % (job_pk, img_name, dataset_pk, traceback.format_exc()))

            except:
                self._log_msg("Failed to post-process predictions generated by job %d for dataset %d!\n%s" % (job_pk, dataset_pk, traceback.format_exc()))

        super()._post_run(template, job, pre_run_success, do_run_success, error)
