import numpy as np
import itertools


class ModelSelector:

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def select(self, *models):
        combinations = self.__get_all_combinations(self.n, self.k)

        scores = self.calc_scores(combinations, *models)

        return combinations[np.argmax(scores)]

    def calc_scores(self, combinations, *models):
        d = {}
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
        return list(itertools.combinations(range(n), k))


class AnyDifferenceSelector(ModelSelector):

    def calc_difference_score(self, *matrices):
        temp = matrices[0]
        for m in matrices[1:]:
            temp = temp + m
        temp = (temp % len(matrices)) > 0
        return sum(sum(temp))


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