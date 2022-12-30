import numpy as np


class Masker:
    def __init__(self, p):
        """
        p: percentage of weights pruned, i.e. p=20, 80% of the weights will remain
        """
        self.p = p
        self.mask_ = None

    def __repr__(self):
        return f"Pruning {self.p}% of weights for weight matrices of shape {self.mask_.shape}"

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        pass

    def apply(self, w):
        if self.mask_ is None:
            raise AttributeError("Can't apply mask that has not yet been calculated")

        return np.multiply(w, self.mask_)


class FMagMasker(Masker):
    def __init__(self, p, msg="Default"):
        super().__init__(p)
        self.msg = msg

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        mag = np.abs(w_f)
        threshold = np.percentile(mag, self.p)
        self.mask_ = np.asarray(mag > threshold, int)
        return self.mask_


class NonZeroMasker(Masker):
    def __init__(self, *args):
        super().__init__(None)

    def mask(self, w_i: np.array = None, w_f: np.array = None) -> np.array:
        self.mask_ = np.asarray(w_f != 0, int)
        return self.mask_
