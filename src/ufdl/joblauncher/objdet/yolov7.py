import re
from dataclasses import dataclass
from glob import glob
import os
import traceback
from typing import Dict, List, Optional, Tuple

from ufdl.jobcontracts.standard import Train, Predict

from ufdl.joblauncher.core.executors import AbstractTrainJobExecutor, AbstractPredictJobExecutor
from ufdl.joblauncher.core.executors.descriptors import ExtraOutput, Parameter
from ufdl.joblauncher.core.executors.parsers import CommandProgressParser
from ufdl.jobtypes.base import Boolean, Integer, String
from ufdl.jobtypes.standard import Name, PK
from ufdl.jobtypes.standard.container import Array

from ufdl.jobtypes.standard.server import Domain, Framework, PretrainedModel, PretrainedModelInstance
from ufdl.jobtypes.standard.util import BLOB

from ufdl.json.object_detection import VideoAnnotation

from ufdl.pythonclient.functional.core.models.pretrained_model import download as pretrainedmodel_download

from wai.json.raw import RawJSONObject

from ..utils import write_to_file
from .core import (
    AbstractObjDetPredictJobExecutor,
    read_rois,
    calculate_confidence_scores,
    store_annotations,
    store_scores
)


DOMAIN_TYPE = Domain("Object Detection")
FRAMEWORK_TYPE = Framework("yolo", "v7")
OBJECT_DETECTION_YOLO_V7_CONTRACT_TYPES = {'DomainType': DOMAIN_TYPE, 'FrameworkType': FRAMEWORK_TYPE}

YOLO_V7_CFG = \
"""
# parameters
nc: ${num-classes}  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0

   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2      
   [-1, 1, Conv, [64, 3, 1]],

   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4  
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 11

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 16-P3/8  
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 24

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 29-P4/16  
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 37

   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 42-P5/32  
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 50
  ]

# yolov7 head
head:
  [[-1, 1, SPPCSPC, [512]], # 51

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [37, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 63

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [24, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 75

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, 63], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 88

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, 51], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 101

   [75, 1, RepConv, [256, 3, 1]],
   [88, 1, RepConv, [512, 3, 1]],
   [101, 1, RepConv, [1024, 3, 1]],

   [[102,103,104], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
"""


class ObjectDetectionTrain_Yolo_v7(AbstractTrainJobExecutor):
    """
    For executing Tensorflow object detection jobs.
    """
    _cls_contract = Train(OBJECT_DETECTION_YOLO_V7_CONTRACT_TYPES)

    pretrained_model: PretrainedModelInstance = Parameter(
        PK(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE)),
        Name(PretrainedModel(DOMAIN_TYPE, FRAMEWORK_TYPE))
    )

    image_size: int = Parameter(
        Integer()
    )

    epochs: int = Parameter(
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
        self.progress(0.05, comment="Downloading dataset...")
        output_dir = self.job_dir + "/data"
        self._download_dataset(pk, output_dir)

        # determine number of classes
        self.progress(0.1, comment="Reading labels...")
        self.class_labels = []
        with open(os.path.join(self.job_dir, "data", "labels.csv")) as lf:
            lines = lf.readlines()
        for line in lines[1:]:
            index, label = line.strip().split(",", 1)
            self.class_labels.append(label)
        self.log_msg(f"{len(self.class_labels)} labels: {self.class_labels}")

        # download pretrained model and put it into models dir
        self.progress(0.15, comment="Downloading pre-trained model...")
        model_file = self.job_dir + "/models/pretrained.pt"
        with open(model_file, "wb") as mf:
            for b in pretrainedmodel_download(self.context, self.pretrained_model.pk):
                mf.write(b)

        # replace classes in config and save it to disk
        self.progress(0.17, comment="Generating config file...")
        config_code = YOLO_V7_CFG.replace("${num-classes}", str(len(self.class_labels)))
        config_file = os.path.join(self.job_dir, "models", "yolov7.yaml")
        with open(config_file, "w") as tf:
            tf.write(config_code)
        self.log_file("Config code:", config_file)

        # replace parameters in template and save it to disk
        self.progress(0.2, comment="Generating training template...")
        template_code = self._expand_template({
            "num-classes": len(self.class_labels),
            "classes": self.class_labels
        })
        if not isinstance(template_code, str):
            template_code = "\n".join(template_code)
        template_file = os.path.join(self.job_dir, "data", "dataset.yaml")
        with open(template_file, "w") as tf:
            tf.write(template_code)
        self.log_file("Template code:", template_file)

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
                image_args=[
                    f"yolov7_train",
                    f"--img={self.image_size}",
                    f"--batch-size=16",
                    f"--epochs={self.epochs}",
                    f"--data=/data/dataset.yaml",
                    f"--cfg=/models/yolov7.yaml",
                    f"--weights=/models/pretrained.pt",
                    f"--project=/output",
                    f"--name=job-number-{self.job_pk}"
                ],
                command_progress_parser=YoloTrainCommandProgressParser()
            )
        )

    def _post_run(
            self,
            pre_run_success: bool,
            do_run_success: bool,
            error: Optional[str]
    ):
        if do_run_success:
            self.progress(0.9, comment="Exporting model as ONNX...")

            # export model
            self._fail_on_error(
                self._run_image(
                    self.docker_image.url,
                    docker_args=["--shm-size", "8G"],
                    volumes=[
                        self.job_dir + "/data" + ":/data",
                        self.job_dir + "/models" + ":/models",
                        self.job_dir + "/output" + ":/output",
                    ],
                    image_args=[
                        f"yolov7_export",
                        f"--weights", "/output/job-number-{self.job_pk}/weights/best.pt",
                        "--grid",
                        "--end2end",
                        "--simplify",
                        "--topk-all", str(len(self.class_labels)),
                        f"--img-size", str(self.image_size), str(self.image_size),
                        f"--max-wh", str(self.image_size),
                    ]
                )
            )

            self.progress(0.95, comment="Uploading model...")

            # zip+upload exported model
            zipfile = self.job_dir + "/model.zip"
            self._compress(
                [
                    f"{self.job_dir}/output/job-number-{self.job_pk}/weights/best.pt",
                    f"{self.job_dir}/output/job-number-{self.job_pk}/weights/best.onnx",
                    f"{self.job_dir}/data/labels.csv"
                ],
                zipfile,
                strip_path=True
            )
            self._upload(self.contract.model, zipfile)

        super()._post_run(pre_run_success, do_run_success, error)


class ObjectDetectionPredict_Yolo_v7(AbstractObjDetPredictJobExecutor):
    """
    For executing Tensorflow object detection prediction jobs.
    """
    _cls_contract = Predict(OBJECT_DETECTION_YOLO_V7_CONTRACT_TYPES)

    image_size: int = Parameter(
        Integer()
    )

    store_predictions: bool = Parameter(Boolean())
    confidence_scores: Tuple[str, ...] = Parameter(Array(String()))

    predictions_csv = ExtraOutput(BLOB("csv"))

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
        model = self.job_dir + "/model.zip"
        with open(model, "wb") as zip_file:
            write_to_file(zip_file, self[self.contract.model])

        # decompress model
        output_dir = self.job_dir + "/output"
        msg = self._decompress(model, output_dir)
        if msg is not None:
            raise Exception("Failed to extract model pk=%d!\n%s" % (pk, msg))

        # determine number of classes
        self.class_labels = []
        with open(os.path.join(output_dir, "labels.csv")) as lf:
            lines = lf.readlines()
        for line in lines[1:]:
            index, label = line.strip().split(",", 1)
            self.class_labels.append(label)
        self.log_msg(f"{len(self.class_labels)} labels: {self.class_labels}")

        # replace parameters in template and save it to disk
        template_code = self._expand_template({
            "num-classes": len(self.class_labels),
        })
        if not isinstance(template_code, str):
            template_code = "\n".join(template_code)
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
                docker_args=["--shm-size", "8G"],
                volumes=[
                    f"{self.job_dir}/data:/data",
                    f"{self.job_dir}/models:/models",
                    f"{self.job_dir}/output:/output",
                    f"{self.job_dir}/prediction:/prediction"
                ],
                image_args=[
                    f"yolov7_predict_poll",
                    f"--model=/output/best.pt",
                    f"--no_trace",
                    f"--confidence_threshold", self._expand_parameters("${confidence-threshold}"),
                    f"--iou_threshold", self._expand_parameters("${iou-threshold}"),
                    f"--image_size={self.image_size}",
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
        dataset_pk: int = self.dataset.pk

        # zip+upload predictions
        if do_run_success:
            self._compress_and_upload(
                self.predictions_csv,
                glob(self.job_dir + "/prediction/out/*.csv"),
                self.job_dir + "/predictions.zip")

        # post-process predictions
        if do_run_success and self.store_predictions:
            try:
                # Create a buffer to accumulate video annotations into
                video_annotations: Dict[str, List[VideoAnnotation]] = {}

                for f in glob(self.job_dir + "/prediction/out/*"):
                    if f.endswith(".csv") or f.endswith("-mask.png"):
                        continue
                    img_name = os.path.basename(f)
                    # load CSV file and create annotations
                    csv_file = os.path.splitext(f)[0] + "-rois.csv"
                    annotations, scores = read_rois(csv_file)

                    # See if the image filename comes from a video frame
                    parsed_video_frame_filename = ParsedVideoFrameFilename.try_parse_from_string(img_name)
                    if parsed_video_frame_filename is not None:
                        # Get the list of video annotations for the video
                        video_annotations_for_video = video_annotations.get(parsed_video_frame_filename.video_filename)

                        # If there is no list yet, create it
                        if video_annotations_for_video is None:
                            video_annotations_for_video = []
                            video_annotations[parsed_video_frame_filename.video_filename] = video_annotations_for_video

                        # Add the video annotations for this frame
                        video_annotations_for_video += [
                            VideoAnnotation.from_image_annotation(annotation, parsed_video_frame_filename.frame_time)
                            for annotation in annotations
                        ]

                    else:
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

                # Store the video annotations back into the dataset
                # TODO: Implement scores/confidence scores for videos as well
                for video_filename, video_annotations_for_video in video_annotations.items():
                    store_annotations(self, dataset_pk, video_filename, video_annotations_for_video)

            except:
                self.log_msg(
                    f"Failed to post-process predictions generated by job {self.job_pk} for dataset {dataset_pk}!\n"
                    f"{traceback.format_exc()}"
                )

        super()._post_run(pre_run_success, do_run_success, error)


# Regex for matching the format filename of an image which is actually a video frame
VIDEO_FRAME_FILENAME_REGEX = re.compile(
    r"""
    ^                                   # Regex start
    (?P<video_filename>.*)              # The filename of the source video
    @                                   # Literal @
    \|                                  # Opening |
    frametime                           # Keyword 'frametime'
    =                                   # Literal =
    (?P<frametime>                      # Frametime (float)
        ([0-9]+(\.[0-9]*)?)             # Either one or more leading digits, optionally followed by a fractional part,
        |                               # or,
        (\.[0-9]+)                      # just a fractional part (dot followed by one or more digits)
    )
    \|                                  # Closing |
    \.jpg                               # .jpg extension
    $                                   # Regex end
    """,
    flags=re.VERBOSE
)


@dataclass
class ParsedVideoFrameFilename:
    """
    Represents the results of parsing a video-frame filename.
    """
    # The source filename
    video_frame_filename: str

    # The original filename of the video
    video_filename: str

    # The timestamp from which the frame image was taken
    frame_time: float

    @classmethod
    def try_parse_from_string(cls, string: str) -> Optional['ParsedVideoFrameFilename']:
        match = VIDEO_FRAME_FILENAME_REGEX.match(string)

        if match is None:
            return None

        return ParsedVideoFrameFilename(
            match[0],
            match[1],
            float(match[2])
        )


# Regex for parsing the progress command output
YOLO_V7_TRAIN_COMMAND_OUTPUT_REGEX = re.compile(
    r"""
    ^                           # Regex start
    .*?                         # Some leading characters
    (?P<epoch_current>\d+)      # The current epoch (0-based)...
    /                           # ...out of...
    (?P<epoch_out_of>\d+)       # ...how many epochs (also 0-based)
    .*?                         # Some intervening characters
    (?P<batch_current>\d+)      # The position of the batch in the current epoch (0-based)...
    /                           #...out of...
    (?P<batch_out_of>\d+)       #...how many batches (also 1-based)
    .*                          # Some trailing characters
    $                           # Regex end
    """,
    flags=re.VERBOSE
)


class YoloTrainCommandProgressParser(CommandProgressParser):
    def parse(self, cmd_output: str, last_progress: float) -> Tuple[float, Optional[RawJSONObject]]:
        # See if the output matches the known progress format
        match = YOLO_V7_TRAIN_COMMAND_OUTPUT_REGEX.match(cmd_output)

        # If not, just return the last progress
        if match is None:
            return last_progress, None

        # How many epochs are there? (epoch_out_of is 0-based)
        num_epochs = int(match.group('epoch_out_of')) + 1

        # How many epochs have been completed? (epoch_current is 0-based)
        completed_epochs = int(match.group('epoch_current'))

        # What percentage of epochs have been completed?
        epoch_progress = completed_epochs / num_epochs

        # How many batches are there in this epoch? (batch_out_of is 1-based)
        num_batches = int(match.group('batch_out_of'))

        # How many batches have been completed? (batch_current is 0-based)
        completed_batches = int(match.group('batch_current'))

        # What percentage of the way through the current epoch are we?
        batch_progress = completed_batches / num_batches

        # What percentage of the way through the overall training process are we?
        overall_progress = epoch_progress + batch_progress / num_epochs

        # Train command represents only 70% of the overall job execution
        return overall_progress * 0.7 + 0.2, None
