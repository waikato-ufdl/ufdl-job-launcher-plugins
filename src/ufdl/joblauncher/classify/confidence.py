import math

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

    def calculate(self, label_scores):
        """
        Returns the confidence score calculated from the label/score dictionary.

        :param label_scores: the dictionary with label-score relation.
        :type label_scores: dict
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        raise NotImplementedError()


class TopScore(AbstractConfidenceScore):
    """
    Returns the score of the label with the highest score.
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        return "top_score"

    def calculate(self, label_scores):
        """
        Returns the confidence score calculated from the label/score dictionary.

        :param label_scores: the dictionary with label-score relation.
        :type label_scores: dict
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        result = 0.0
        for l in label_scores:
            score = label_scores[l]
            if score > result:
                result = score
        return {self.name(): result}


class Entropy(AbstractConfidenceScore):
    """
    Returns the entropy calculated as follows:

    E = sum(score_i * ln(score_i))
    """

    def name(self):
        """
        Returns the name of the confidence score.

        :return: the name
        :rtype: str
        """
        return "entropy"

    def calculate(self, label_scores):
        """
        Returns the confidence score calculated from the label/score dictionary.

        :param label_scores: the dictionary with label-score relation.
        :type label_scores: dict
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        try:
            result = 0.0
            for l in label_scores:
                score = label_scores[l]
                result += score * math.log(score)
            result *= -1
        except:
            result = float("NaN")
        return {self.name(): result}


class Common(AbstractConfidenceScore):
    """
    Combines common scores.
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

    def calculate(self, label_scores):
        """
        Returns the confidence score calculated from the label/score dictionary.

        :param label_scores: the dictionary with label-score relation.
        :type label_scores: dict
        :return: the confidence score dictionary (name -> float)
        :rtype: dict
        """
        result = dict()
        self._add(result, TopScore().calculate(label_scores))
        self._add(result, Entropy().calculate(label_scores))
        return result