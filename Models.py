import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import glob
import random
import matplotlib.pyplot as plt


class ModelWrapper:
    def __init__(self, model, initial_weights=None, history=None, loaded=False):
        self.model = model
        self.initial_weights = model.get_weights() if initial_weights is None else initial_weights
        self.loaded = loaded  # True if loaded from a file, False if newly trained
        self.history = history
        self.refit_ = False  # switches to true if the model has been fitted a second time

        self.maskers = None
        self.source_weights = None
        self.re_history = None

    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs)

    def refit(self, *args, **kwargs):
        self.refit_ = True
        self.re_history = self.model.fit(*args, **kwargs)

    def prune(self, pruner, p):
        """
        prunes (100 - p)% of the weights in each layer based on the rules of the pruner and returns the maskers for each layer
        """
        maskers = []
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue
            W = layer.get_weights()[0]

            masker = pruner(p)
            masker.mask(w_f=W)
            pruned_weights = masker.apply(W)
            maskers.append(masker)

            layer.set_weights([pruned_weights])
        self.maskers = maskers
        return maskers

    def reset(self):
        """
        Resets weights to their initial state theta_0
        """
        i = 0
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue
            layer.set_weights([self.maskers[i].apply(self.initial_weights[i])])
            i += 1

    def set_weights(self, weights):
        i = 0
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue

            layer.set_weights([weights[i]])
            i += 1

    @classmethod
    def load(cls, filename):
        m = keras.models.load_model(filename)
        with open(filename.replace("models", "history"), "rb") as file_pi:
            history = pickle.load(file_pi)
            h = keras.callbacks.History()
            h.history = history
        w = []
        for i in range(len(m.get_weights())):
            w.append(np.loadtxt(filename.replace("models", "weights") + "_" + str(i) + ".dat"))
        return ModelWrapper(m, w, h, True)

    def store(self, folder, i):
        self.model.save("models/" + folder + "/" + str(i))

        with open('history/' + folder + "/" + str(i), 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        for layer_i, layer_weights in enumerate(self.initial_weights):
            np.savetxt("weights/" + folder + "/" + str(i) + "_" + str(layer_i) + ".dat", layer_weights)


class ModelIO:
    """
    Provides an API for unpruned models
    """

    def __init__(self, source_model, optimizer, loss, metrics, pruning_percentage, **train_props):
        self.models = []
        self.architecture = source_model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.pruning_percentage = pruning_percentage

        self.train_props = train_props

        self.model_generator = None

    def __new_clone(self):
        m = tf.keras.models.clone_model(self.architecture)
        m.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return m

    def load_models(self, folder: str, count: int):
        """
        load count random models from the models folder
        """

        def get_model_names():
            return glob.glob("models/" + folder + "/*")

        def get_model_count():
            return len(get_model_names())

        def load_n(n: int):
            load_names = random.sample(get_model_names(), n)
            for name in load_names:
                print(folder, name)
                self.models.append(ModelWrapper.load(name))

        def load_all():
            for filename in get_model_names():
                self.models.append(ModelWrapper.load(filename))

        def train_new():
            clone = ModelWrapper(self.__new_clone())
            clone.fit(**self.train_props)
            self.models.append(clone)

        model_count = get_model_count()
        if model_count == count:
            load_all()
        elif model_count >= count:
            load_n(count)
        else:
            new_count = count - model_count

            load_all()
            for i in range(new_count):
                train_new()

        self.store_models("mnist")

    def store_models(self, name: str):
        for i, model in enumerate(self.models):
            if not model.loaded:
                model.store(name, i)

    def __get_model_list(self):
        for m in self.models:
            yield m

    def fetch(self):
        if self.model_generator is None:
            self.model_generator = self.__get_model_list()
        return next(self.model_generator)

    def __get_original_model_histories(self, filter_func=lambda x: True):
        for m in filter(filter_func, self.models):
            yield m.history

    def __get_pruned_model_histories(self, filter_func=lambda x: True):
        for m in filter(filter_func, self.models):
            yield m.re_history

    def plot_history(self, ylim=[0, 1]):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        functions = [self.__get_original_model_histories, self.__get_pruned_model_histories]

        for i, p in enumerate(functions):
            history = [h.history['val_sparse_categorical_accuracy'] for h in p(lambda x: x.refit_)]
            history = np.array(history)
            mean = np.mean(history, axis=0)
            max_ = np.max(history, axis=0)
            min_ = np.min(history, axis=0)

            plt.plot(mean)
            plt.vlines(x=range(self.train_props["epochs"]), ymin=min_, ymax=max_, colors=colors[i])
        plt.legend([100] + self.pruning_percentage, loc='upper left')
        plt.ylim(ylim)
        plt.show()

    class HistoryWrapper:
        def __init__(self, *histories):
            self.histories = np.array(histories)
            self.mean = np.mean(self.histories, axis=0)
            self.max_ = np.max(self.histories, axis=0)
            self.min_ = np.min(self.histories, axis=0)
