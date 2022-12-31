import keras


class FreezeCallback(keras.callbacks.Callback):
    """
    Callback that resets "pruned" weights after each iteration
    """

    def __init__(self, masks):
        self.masks = masks

    def on_train_start(self, batch, logs=None):
        print(self.masks)

    def on_batch_end(self, batch, logs=None):
        i = 0
        for l in self.model.layers:
            if len(l.get_weights()) == 0:
                continue
            l.set_weights([self.masks[i].apply(l.get_weights()[0])])
            i += 1
