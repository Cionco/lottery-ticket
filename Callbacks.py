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


class LotteryCallback:
    """
    Provides an API, which enables a user to check into the progress of an experiment at multiple steps during a lottery
    ticket experiment.
    """
    def __call__(self, *args, **kwargs):
        pass

    def on_lottery_start(self, lt, current_model):
        pass

    def on_lottery_end(self, lt, current_model):
        pass

    def on_iteration_start(self, lt, current_model):
        pass

    def on_iteration_end(self, lt, current_model):
        pass

    def on_prune_start(self, lt, current_model):
        pass

    def on_prune_end(self, lt, current_model):
        pass

    def on_select_start(self, lt, current_model):
        pass

    def on_select_end(self, lt, current_model):
        pass

    def on_reset_start(self, lt, current_model):
        pass

    def on_reset_end(self, lt, current_model):
        pass

    def on_marry_start(self, lt, current_model):
        pass

    def on_marry_end(self, lt, current_model):
        pass

    def on_retrain_start(self, lt, current_model):
        pass

    def on_retrain_end(self, lt, current_model):
        pass

