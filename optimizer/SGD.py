import tensorflow as tf
from keras.optimizers import SGD


class MaskingSGD(SGD):
    def set_masks(self, masks):
        self.masks = masks

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss, tape=tape)
        if hasattr(self, 'masks'):
            grads_and_vars = [
                (tf.multiply(grad, mask), var) for mask, (grad, var) in zip(self.masks, grads_and_vars)
            ]
        return self.apply_gradients(grads_and_vars)