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
        :return: the confidence score
        :rtype: float
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
        :return: the confidence score
        :rtype: float
        """
        result = 0.0
        for l in label_scores:
            score = label_scores[l]
            if score > result:
                result = score
        return result


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
        :return: the confidence score
        :rtype: float
        """
        try:
            result = 0.0
            for l in label_scores:
                score = label_scores[l]
                result += score * math.log(score)
        except:
            result = float("NaN")
        return result

