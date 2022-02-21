import csv
import json
import traceback
from typing import Iterable, List, Tuple

from ufdl.json.object_detection import Annotation, Polygon
from ufdl.joblauncher.core import load_class
from ufdl.joblauncher.core.executors import AbstractJobExecutor
from ufdl.pythonclient.functional.object_detection.dataset import get_metadata, set_metadata, set_annotations_for_image

from .confidence import AbstractConfidenceScore


def read_rois(csv_file: str) -> Tuple[List[Annotation], List[float]]:
    """
    Loads the specified ROIs CSV file and generates a list of Annotation objects
    and a list of scores from it.

    :param csvfile: the CSV file to read
    :return: the tuple of annotations list and scores list
    """
    annotations: List[Annotation] = []
    scores: List[float] = []
    with open(csv_file, "r") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            if ('x' in row) and ('y' in row) and ('w' in row) and ('h' in row) and ('label_str' in row) and ('score' in row):
                polygon = None
                if ('poly_x' in row) and ('poly_y' in row):
                    xs = row['poly_x'].split(",")
                    ys = row['poly_y'].split(",")
                    p = []
                    for x, y in zip(xs, ys):
                        p.append([int(float(x)), int(float(y))])
                    polygon = Polygon(points=p)
                if polygon is not None:
                    annotation = Annotation(
                        x=int(float(row['x'])),
                        y=int(float(row['y'])),
                        width=int(float(row['w'])),
                        height=int(float(row['h'])),
                        label=row['label_str'],
                        polygon=polygon)
                else:
                    annotation = Annotation(
                        x=int(float(row['x'])),
                        y=int(float(row['y'])),
                        width=int(float(row['w'])),
                        height=int(float(row['h'])),
                        label=row['label_str'])
                annotations.append(annotation)
                scores.append(float(row['score']))

    return annotations, scores


def store_annotations(executor, dataset_pk, img_name, annotations):
    """
    Stores the annotations in the backend.

    :param executor: the executor class this is done for
    :type executor: AbstractJobExecutor
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :type dataset_pk: int
    :param img_name: the name of the image the scores were calculated for
    :type img_name: str
    :param annotations: the list of Annotation objects
    :type annotations: list[Annotation]
    """
    # set annotations for image
    try:
        set_annotations_for_image(executor.context, dataset_pk, img_name, annotations)
    except:
        executor.log_msg("Failed to add annotations generated by job %d to dataset %d!\n%s" % (executor.job_pk, dataset_pk, traceback.format_exc()))


def store_scores(executor, dataset_pk, img_name, scores):
    """
    Stores the annotations in the backend.

    :param executor: the executor class this is done for
    :type executor: AbstractJobExecutor
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :type dataset_pk: int
    :param img_name: the name of the image the scores were calculated for
    :type img_name: str
    :param scores: the list of float scores
    :type scores: list[float]
    """
    try:
        metadata = get_metadata(executor.context, dataset_pk, img_name)
        if metadata == "":
            metadata = dict()
        else:
            metadata = json.loads(metadata)
        metadata['scores'] = scores
        set_metadata(executor.context, dataset_pk, img_name, json.dumps(metadata))
    except:
        executor.log_msg("Failed to add scores of job %d for image %s in dataset %d!\n%s" % (executor.job_pk, img_name, dataset_pk, traceback.format_exc()))


def calculate_confidence_scores(
        executor: AbstractJobExecutor,
        dataset_pk: int,
        img_name: str,
        confidence_score_classes: Iterable[str],
        annotations: List[Annotation],
        scores: List[float]
):
    """
    Calculates and stores confidence scores.

    :param executor: the executor class this is done for
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :param img_name: the name of the image the scores were calculated for
    :param confidence_score_classes: the list of class names
    :param annotations: the list of annotations to use for the score confidence calculation
    :param scores: the list of scores to use for the confidence score calculation
    """

    # instantiate calculators
    conf_score_obj: List[AbstractConfidenceScore] = []
    for c in confidence_score_classes:
        try:
            cls = load_class(c)
            if not issubclass(cls, AbstractConfidenceScore):
                executor.log_msg(f"Confidence score class '{c}' does not sub-class {AbstractConfidenceScore.__qualname__}")
                continue
            conf_score_obj.append(cls())
        except:
            executor.log_msg(
                f"Failed to instantiate confidence score class: {c}\n"
                f"{traceback.format_exc()}"
            )

    # calculate the scores
    if len(conf_score_obj) > 0:
        try:
            conf_scores = dict()
            for c in conf_score_obj:
                current = c.calculate(annotations, scores)
                for k in current:
                    conf_scores[k] = current[k]
            metadata = get_metadata(executor.context, dataset_pk, img_name)
            if metadata == "":
                metadata = dict()
            else:
                metadata = json.loads(metadata)
            metadata['confidence'] = conf_scores
            set_metadata(executor.context, dataset_pk, img_name, json.dumps(metadata))
        except:
            executor.log_msg(
                f"Failed to add confidence scores of job {executor.job_pk} "
                f"for image {img_name} in dataset {dataset_pk}!\n"
                f"{traceback.format_exc()}"
            )
