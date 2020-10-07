import csv
from ufdl.json.object_detection import Annotation


def rois_to_annotations(csv_file):
    """
    Loads the specified CSV file and generates a list of Annotation objects
    and a list of scores from it.

    :param csvfile: the CSV file to read
    :type csvfile: str
    :return: the tuple of annotations list and scores list
    :rtype: tuple
    """
    annotations = []
    scores = []
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

    return annotations, scores
