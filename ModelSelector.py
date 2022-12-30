import numpy as np
import itertools
from lottery.Models import ModelWrapper


class ModelSelector:

    def __init__(self, n, k="auto"):
        """
        :param n: how many models the selector gets to chose from
        :param k: how many models the selector should select. Can be any integer value or "auto".
        """
        self.n = n
        self.k = k

    def select(self, *models):
        """
        Gets a tuple of ModelWrapper objects as input and selects k of them based on rules defined in subclasses

        :param models: tuple of n ModelWrappers

        :returns:       list of ModelWrappers that are selected
        """

        combinations = self.__get_all_combinations(self.n, self.k)

        weight_masks = [m.model.get_weights() for m in models]
        scores = self.calc_scores(combinations, *weight_masks)

        selected_combination = combinations[np.argmax(scores)]
        return list(map(models.__getitem__, selected_combination))

    def calc_scores(self, combinations, *models) -> [int]:
        """
        Calculates a score for each combination of models and returns a list of scores in the same order as the
        combinations were passed

        :param combinations:  list of touples encoding one combination each. e.g. n=3 k=2 ->
                                [(1, 2), (1, 3), (2, 3)]
        :param models:        touple of lists with weights masks

        :returns:             list of scores
        """
        d = {}
        if len(combinations) == 1:
            return [1]
        for c in combinations:  # for each combination
            for i, m in enumerate(zip(*models)):  # for each "layer"
                if i not in d:
                    d[i] = []
                d[i].append(self.calc_difference_score(*list(map(m.__getitem__, c))))

        total_scores = [sum(x) for x in zip(*d.values())]
        return total_scores

    def calc_difference_score(self, *matrices):
        pass

    def __get_all_combinations(self, n, k):
        if k == "auto":
            k = n
        return list(itertools.combinations(range(n), k))


class AnyDifferenceSelector(ModelSelector):
    """
    Assigns a difference score of 1 to any place in a mask where there's at least one difference along the masks
    """

    def __init__(self, n: int, k: int):
        super().__init__(n, k)

    def calc_difference_score(self, *matrices):
        temp = matrices[0]
        for m in matrices[1:]:
            temp = temp + m
        temp = (temp % len(matrices)) > 0
        return sum(sum(temp))


class AllModelsSelector(ModelSelector):
    """
    Selects all models that are passed in
    """

    def __init__(self, n: int = 1):
        super().__init__(n, "auto")


if __name__ == "__main__":
    a = [np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [1, 1, 0, 1],
                   [0, 1, 1, 1]]),
         np.array([[1, 0, 0],
                   [0, 1, 1],
                   [0, 1, 1]]),
         np.array([[1, 0],
                   [1, 0]])
         ]

    b = [np.array([[0, 1, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 0]]),
         np.array([[1, 0, 1],
                   [1, 1, 1],
                   [0, 1, 1]]),
         np.array([[1, 1],
                   [1, 1]])
         ]

    c = [np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 1, 0]]),
         np.array([[1, 0, 0],
                   [1, 0, 1],
                   [1, 0, 0]]),
         np.array([[1, 0],
                   [1, 0]])
         ]

    w = [a, b, c]

    combination = AnyDifferenceSelector(3, 2).select(a, b, c)

    print(combination)
