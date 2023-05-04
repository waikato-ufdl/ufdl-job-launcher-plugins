import csv
import json
import traceback
from typing import Dict, Iterable, List, Tuple

from ufdl.joblauncher.core import load_class
from ufdl.joblauncher.core.executors import AbstractJobExecutor
from ufdl.pythonclient.functional.image_classification.dataset import get_metadata
from ufdl.pythonclient.functional.image_classification.dataset import set_metadata

from .confidence import AbstractConfidenceScore


def read_scores(csv_file: str) -> Tuple[Dict[str, float], str]:
    """
    Reads the labels and scores from the CSV file.

    :param csv_file: the CSV file to read
    :return: tuple of the dictionary with label -> score relations and the top label
    """
    label = "?"
    prob = 0.0
    label_scores: Dict[str, float] = dict()
    with open(csv_file, "r") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            if ('probability' in row) and ('label' in row):
                label_scores[row['label']] = float(row['probability'])
                if float(row['probability']) > prob:
                    prob = float(row['probability'])
                    label = row['label']
    return label_scores, label


def calculate_confidence_scores(
        executor: AbstractJobExecutor,
        dataset_pk: int,
        img_name: str,
        confidence_score_classes: Iterable[str],
        label_scores: Dict[str, float]
):
    """
    Calculates and stores confidence scores.

    :param executor: the executor class this is done for
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :param img_name: the name of the image the scores were calculated for
    :param confidence_score_classes: the list of class names
    :param label_scores: the labels with their associated scores
    """

    # instantiate calculators
    conf_score_obj: List[AbstractConfidenceScore] = []
    for c in confidence_score_classes:
        try:
            cls = load_class(c)
            if not issubclass(cls, AbstractConfidenceScore):
                #executor.log_msg(f"Confidence score class '{c}' does not sub-class {AbstractConfidenceScore.__qualname__}")
                executor.log_msg(f"Confidence score class '{c}' -> '{str(type(cls))}' does not sub-class {str(type(AbstractConfidenceScore))}")
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
                current = c.calculate(label_scores)
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
