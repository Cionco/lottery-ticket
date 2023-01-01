import numpy as np


class Masker:
    """
    A masker learns a mask from one or two weight matrices.
    This learned mask can then be applied to other matrices.

    Masks are np.arrays with only 0 and 1 values. "applying" a mask
    means multiplying some matrix with the mask.
    """

    def __init__(self, p):
        """
        p: percentage of weights pruned, i.e. p=20, 80% of the weights will remain
        """
        self.p = p
        self.mask_ = None

    def __repr__(self):
        return f"Pruning {self.p}% of weights for weight matrices of shape {self.mask_.shape}"

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.p)

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        pass

    def apply(self, w):
        if self.mask_ is None:
            raise AttributeError("Can't apply mask that has not yet been calculated")

        return np.multiply(w, self.mask_)


class FMagMasker(Masker):
    """
    This masker selects based on the final weights magnitude.
    Every weight that is below the threshold of the p-percentile will be masked as a 0, the rest as 1
    """
    def __init__(self, p):
        super().__init__(p)

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        mag = np.abs(w_f)
        threshold = np.percentile(mag, self.p)
        self.mask_ = np.asarray(mag > threshold, int)
        return self.mask_


class NonZeroMasker(Masker):
    """
    This masker masks every value that's not 0 as 1.
    """
    def __init__(self, *args):
        super().__init__(None)

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        self.mask_ = np.asarray(w_f != 0, int)
        return self.mask_
