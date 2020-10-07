import csv
import json
import traceback
from ufdl.joblauncher import load_class
from ufdl.joblauncher import AbstractJobExecutor
from ufdl.pythonclient.functional.image_classification.dataset import get_metadata
from ufdl.pythonclient.functional.image_classification.dataset import set_metadata


def read_scores(csv_file):
    """
    Reads the labels and scores from the CSV file.

    :param csv_file: the CSV file to read
    :type csv_file: str
    :return: tuple of the dictionary with label -> score relations and the top label
    :rtype: tuple
    """
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
    return label_scores, label


def calculate_confidence_scores(executor, job_pk, dataset_pk, img_name, confidence_score_classes, label_scores):
    """
    Calcualtes and stores confidence scores.

    :param executor: the executor class this is done for
    :type executor: AbstractJobExecutor
    :param job_pk: the PK of the job being executed
    :type job_pk: int
    :param dataset_pk: the PK of the dataset these scores are calculated for
    :type dataset_pk: int
    :param img_name: the name of the image the scores were calculated for
    :type img_name: str
    :param confidence_score_classes: the list of class names
    :type confidence_score_classes: list
    :param label_scores: the labels with their associated scores
    :type label_scores: dict
    """

    # instantiate calculators
    conf_score_obj = []
    try:
        for c in confidence_score_classes:
            conf_score_obj.append(load_class(c)())
    except:
        executor.log_msg("Failed to instantiate confidence score classes: %s\n%s" % (str(confidence_score_classes), traceback.format_exc()))

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
            executor.log_msg("Failed to add confidence scores of job %d for image %s in dataset %d!\n%s" % (job_pk, img_name, dataset_pk, traceback.format_exc()))
