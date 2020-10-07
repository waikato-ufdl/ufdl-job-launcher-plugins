import numpy as np
from ufdl.json.object_detection import Annotation


class AbstractConfidenceScore(object):
    """
    Ancestor for classes calculating confidence scores.
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        raise NotImplementedError()

    def calculate(self, annotations, scores):
        """
        Returns the confidence score calculated from the annotations list.

        :param annotations: the list of Annotation objects
        :type annotations: list[Annotation]
        :param scores: the list of scores
        :type scores: list[float]
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        raise NotImplementedError()


class ScoreDist(AbstractConfidenceScore):
    """
    Returns the distribution of the scores (min/max/mean/median/stdev).
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        return "score-dist"

    def calculate(self, annotations, scores):
        """
        Returns the confidence score calculated from the annotations list.

        :param annotations: the list of Annotation objects
        :type annotations: list[Annotation]
        :param scores: the list of scores
        :type scores: list[float]
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        result = dict()
        np_scores = np.array(scores)
        result[self.name() + "-min"] = float(np.min(np_scores))
        result[self.name() + "-max"] = float(np.max(np_scores))
        result[self.name() + "-mean"] = float(np.mean(np_scores))
        result[self.name() + "-median"] = float(np.median(np_scores))
        result[self.name() + "-stdev"] = float(np.std(np_scores))
        return result


class ObjectDims(AbstractConfidenceScore):
    """
    Returns the distribution of the object dimensions (min/max/mean/median/stdev) for width/height.
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        return "object-dims"

    def _calculate(self, all, prefix, values):
        """
        Calculates min/max/mean/median/stdev from the values.

        :param all: the dictionary to store the stats in
        :type all: dict
        :param prefix: the prefix to use for the stats names
        :type prefix: str
        :param values: the values to calculate the stats from
        :type values: list
        """
        np_values = np.array(values)
        all[prefix + "-min"] = float(np.min(np_values))
        all[prefix + "-max"] = float(np.max(np_values))
        all[prefix + "-mean"] = float(np.mean(np_values))
        all[prefix + "-median"] = float(np.median(np_values))
        all[prefix + "-stdev"] = float(np.std(np_values))

    def calculate(self, annotations, scores):
        """
        Returns the confidence score calculated from the annotations list.

        :param annotations: the list of Annotation objects
        :type annotations: list[Annotation]
        :param scores: the list of scores
        :type scores: list[float]
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        result = dict()
        widths = []
        heights = []
        for a in annotations:
            widths.append(a.width)
            heights.append(a.height)
        self._calculate(result, "widths", widths)
        self._calculate(result, "heights", heights)
        return result


class Common(AbstractConfidenceScore):
    """
    Returns common scores.
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        return "common"

    def _add(self, all, current):
        """
        Adds the scores from "current" to "all".

        :param all: the dictionary with all the combined scores
        :type all: dict
        :param current: the dictionary of scores to add
        :type current: dict
        """
        for k in current:
            all[k] = current[k]

    def calculate(self, annotations, scores):
        """
        Returns the confidence score calculated from the annotations list.

        :param annotations: the list of Annotation objects
        :type annotations: list[Annotation]
        :param scores: the list of scores
        :type scores: list[float]
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        result = dict()
        self._add(result, ScoreDist().calculate(annotations, scores))
        self._add(result, ObjectDims().calculate(annotations, scores))
        return result
