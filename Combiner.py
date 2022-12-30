from lottery.Masker import *

class Combiner:
    def __init__(self, c, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        """
        c is the amount of lottery tickets that should be combined. If c = 2, 2 networks shall be pretrained and then combined
        """
        self.c = c
        self.mask_combine_type = mask_combine_type
        self.combine_cutoff = combine_cutoff  # maximum percentage of weights to be set after combining

    def marry(self, *marry_weights) -> list:
        """
        Gets the weights of all layers of multiple models and calls marry_layer 
        for each layer
        Returns a list of np arrays and a list of Maskers. Each one are the
        weights of one layer or the Maskers of one layer respectively
        """
        weights = []
        maskers = []
        for t in list(zip(*marry_weights)):
            weights.append(self.marry_layer(*t))

        for w in weights:
            masker = self.mask_combine_type(self.combine_cutoff)
            masker.mask(w_f=w)
            maskers.append(masker)
        return weights, maskers

    def marry_layer(self, *marry_weights) -> np.array:
        pass


class DefaultCombiner(Combiner):
    def __init__(self):
        super().__init__(c=1)

    def marry_layer(self, *marry_weights) -> np.array:
        return marry_weights[0]


def max_mag(a, b):
    abs_a = np.abs(a)
    abs_b = np.abs(b)

    mask_a = np.asarray(abs_a >= abs_b, int)
    mask_b = np.asarray(abs_b > abs_a, int)

    return a * mask_a + b * mask_b


class MaxCombiner(Combiner):

    def __init__(self, c, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(c, mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        temp = marry_weights[0]
        for i in marry_weights[1:]:
            temp = max_mag(temp, i)
        return temp
