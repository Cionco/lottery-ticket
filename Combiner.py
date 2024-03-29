from lottery.Masker import *


class Combiner:
    """
    A combiner takes in one or more models and combines them in some sort of way.
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        """
        :param mask_combine_type: Defines the type of the masker that should be applied to the new weight matrix
                                    after combination. The NonZeroMasker is good to just keep the weight matrix as is
                                    after combination, however one might want to keep the combined model pruned.
                                    e.g. Combining 2 79% pruned networks yields somthing like a 60% pruned network.
                                    When combining 4 79% pruned networks, the result is oftentimes back to using almost
                                    70% of the available weights (less than 40% pruning)
        :param combine_cutoff:    When a "cutoff masker" is used, this defines how much of the combined matrix will be
                                    pruned. So if we're using mask_combine_type=FMagMasker with a combine_cutoff=79,
                                    the resulting weight matrix will have 79% zeros.
        """
        self.mask_combine_type = mask_combine_type
        self.combine_cutoff = combine_cutoff  # maximum percentage of weights to be set after combining

    def __str__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.mask_combine_type.__name__, self.combine_cutoff)

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

        for i, w in enumerate(weights):
            masker = self.mask_combine_type(self.combine_cutoff)
            masker.mask(w_f=w)
            maskers.append(masker)
            weights[i] = masker.apply(w)
        return weights, maskers

    def marry_layer(self, *marry_weights) -> np.array:
        pass


class DefaultCombiner(Combiner):
    """
    The Default combiner is just a Combiner implementation for default cases where no combining should happen.
    It just returns the weights of the first model
    """

    def __init__(self):
        super().__init__()

    def marry_layer(self, *marry_weights) -> np.array:
        return marry_weights[0]


def max_mag(a, b):
    """
    Elementwise, the result will have the value with the higher magnitude.
    If magnitudes are equal, the value from a is kept
        [1, 2, 3, -4, 4],
        [0, -5, 5, 4, -4]
        -----------------
    --> [1, -5, 5, -4, 4]
    """
    abs_a = np.abs(a)
    abs_b = np.abs(b)

    mask_a = np.asarray(abs_a >= abs_b, int)
    mask_b = np.asarray(abs_b > abs_a, int)

    return a * mask_a + b * mask_b


class MaxMagCombiner(Combiner):
    """
    The MaxMagCombiner keeps the weight with the highest magnitude on each position
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        temp = marry_weights[0]
        for i in marry_weights[1:]:
            temp = max_mag(temp, i)
        return temp


class MaxCombiner(Combiner):
    """
    keeps the weight with the highest value, even if that removes a connection
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        temp = marry_weights[0]
        for i in marry_weights[1:]:
            temp = np.maximum(temp, i)
        return temp


class MaxWithoutZeroCombiner(Combiner):
    """
    keeps the weight with the highest value if multiple links are present. If only one link is present, that is always kept
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        temp = marry_weights[0]
        for i in marry_weights[1:]:
            max_ = np.maximum(temp, i)
            min_ = np.minimum(temp, i)

            mask = np.asarray(max_ == 0, int)

            temp = max_ + np.multiply(min_, mask)
        return temp


class MinWithoutZeroCombiner(Combiner):
    """
    keeps the weight with the lowest value if multiple links are present. If only one link is present, that is always kept
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        temp = marry_weights[0]
        for i in marry_weights[1:]:
            max_ = np.maximum(temp, i)
            min_ = np.minimum(temp, i)

            mask = np.asarray(min_ == 0, int)

            temp = min_ + np.multiply(max_, mask)
        return temp


class AvgCombiner(Combiner):
    """
        The AvgCombiner takes the average of all combined weights
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        return np.mean(marry_weights, axis=0)


class AvgNoZeroCombiner(Combiner):
    """
    Takes the average of all weights, disregarding 0, i.e. disconnected connections
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        weight_array = np.asarray(marry_weights, float)
        weight_array[weight_array == 0] = np.nan
        return np.nan_to_num(np.nanmean(weight_array, axis=0))


class AddCombiner(Combiner):
    """
        The AddCombiner sums up all combined weights
    """

    def __init__(self, mask_combine_type=NonZeroMasker, combine_cutoff=None):
        super().__init__(mask_combine_type, combine_cutoff)

    def marry_layer(self, *marry_weights) -> np.array:
        return np.sum(marry_weights, axis=0)
