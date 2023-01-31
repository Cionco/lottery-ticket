import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import glob
import random

from lottery.optimizer.SGD import MaskingSGD


class ModelWrapper:
    """
    Wrapper for model objects and some data on the model.
    Arguments:
        model           -- The tensorflow model
        initial_weights -- the initial weights of the model when it was first compiled
        loaded          -- boolean flag indicating if this model was just trained or loaded from a file
        history         -- the original training history (keras.callbacks.History) of the model
        refit_          -- boolean flag indicating if this model has gone through a second training process,
                            i.e. if it's a final lottery ticket model
        maskers         -- list of maskers that should be applied to the model.
        source_weights  -- if this is a combined model, source weights stores the weight matrices that were combined
        re_history      -- the training history (keras.callbacks.History) of the pruned (& combined) model
    """

    def __init__(self, model, initial_weights=None, history=None, loaded=False):
        self.model = model
        self.initial_weights = model.get_weights() if initial_weights is None else initial_weights
        self.loaded = loaded  # True if loaded from a file, False if newly trained
        self.history = history
        self.refit_ = False  # switches to true if the model has been fitted a second time

        self.maskers = None
        self.masks = None
        self.source_weights = None
        self.re_history = None

    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs, verbose=0)

    def refit(self, *args, **kwargs):
        self.refit_ = True
        self.model.optimizer.set_masks([m.mask_ for m in self.maskers])
        self.re_history = self.model.fit(*args, **kwargs)

    def prune(self, pruner, p):
        """
        prunes p% of the weights in each layer based on the rules of the pruner and returns the maskers for each layer
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
        Resets weights to their initial state theta_0 while setting all pruned weights to 0
        """
        i = 0
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue
            layer.set_weights([self.maskers[i].apply(self.initial_weights[i])])
            i += 1

    def set_weights(self, weights):
        """
        Sets the weights of this model

        :param weights: List of New weights matrices for this model. Each list element is the matrix for one layer
        """
        i = 0
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue

            layer.set_weights([weights[i]])
            i += 1

    def set_maskers(self, maskers):
        self.maskers = maskers


    @classmethod
    def load(cls, filename):
        """
        Creates a new ModelWrapper object by loading model, original training history and initial weights from files.

        :param filename: relative path to the model file
        """
        m = keras.models.load_model(filename, custom_objects={"MaskingSGD": MaskingSGD})
        with open(filename.replace("models", "history"), "rb") as file_pi:
            history = pickle.load(file_pi)
            h = keras.callbacks.History()
            h.history = history
        w = []
        for i in range(len(m.get_weights())):
            w.append(np.loadtxt(filename.replace("models", "weights") + "_" + str(i) + ".dat"))
        return ModelWrapper(m, w, h, True)

    def store(self, folder, i):
        """
        Stores the model, its original training history and its inital weights in seperate files.
        """
        self.model.save("models/" + folder + "/" + str(i))

        with open('history/' + folder + "/" + str(i), 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        for layer_i, layer_weights in enumerate(self.initial_weights):
            np.savetxt("weights/" + folder + "/" + str(i) + "_" + str(layer_i) + ".dat", layer_weights)


class ModelIO:
    """
    Provides an API for unpruned models

    Attributes:
        models              -- list of all loaded ModelWrapper objects
        architecture        -- untrained tensorflow model with the desired architecture
        optimizer           -- tensorflow optimizer that should be used for model training
        loss                -- tensorflow loss that should be used for model training
        metrics             -- list of metrics to be evaluated by the model during training and testing
        pruning_percentage  -- how many weights should be pruned
        train_props         -- all arguments needed for training, like epochs, training and validation data, batchsize etc
        model_generator     -- generator object for models. Only to be used by the fetch method.
    """

    def __init__(self, source_model, optimizer, loss, metrics, pruning_percentage, **train_props):
        #self.models = []
        self.architecture = source_model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.pruning_percentage = pruning_percentage

        self.train_props = train_props

        self.model_generator = None

    def __new_clone(self):
        """
        Creates and compiles a new clone of the model
        """
        m = tf.keras.models.clone_model(self.architecture)
        m.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return m

    def load_models(self, folder, count):
        """
        Initiates the model loader generator by calling __load_model_generator
        Returns nothing but sets the self.model_generator attribute

        :param folder: something like the project name. folder structure is
                        models/<projectname>/model_files
                        history/<projectname>/history_files
                        weights/<projectname>/weight_files
        :param count:  how many models should be loaded
        """
        self.model_generator = self.__load_model_generator(folder, count)

    def __load_model_generator(self, folder: str, count: int):
        """
        load count models. Depending on how many models are already saved in folders (saved):
            saved > count : load n random models from files
            saved == count: load all models from files
            saved < count : load all models from files and then create and train count - saved new models
                            new models are then saved
        Returns a generator that loads the models when they're needed

        :param folder: something like the project name. folder structure is
                        models/<projectname>/model_files
                        history/<projectname>/history_files
                        weights/<projectname>/weight_files
        :param count:  how many models should be loaded

        """

        def get_model_names():
            return glob.glob("models/" + folder + "/*")

        def get_model_count():
            return len(get_model_names())

        def load_n(n: int):
            load_names = random.sample(get_model_names(), n)
            for name in load_names:
                print(folder, name)
                #self.models.append(ModelWrapper.load(name))
                yield ModelWrapper.load(name)

        def load_all():
            for filename in get_model_names():
                #self.models.append(ModelWrapper.load(filename))
                yield ModelWrapper.load(filename)

        def train_new():
            clone = ModelWrapper(self.__new_clone())
            clone.fit(**self.train_props)
            #self.models.append(clone)
            return clone

        model_count = get_model_count()
        if model_count == count:
            yield from load_all()
        elif model_count >= count:
            yield from load_n(count)
        else:
            new_count = count - model_count

            load_all()
            for i in range(new_count):
                new_model = train_new()
                new_model.store(folder, i + model_count)
                yield new_model

    @DeprecationWarning
    def store_models(self, name: str):
        """
        Stores all models that just trained
        """
        for i, model in enumerate(self.models):
            if not model.loaded:
                model.store(name, i)

    def fetch(self):
        """
        Get the next model for the experiment
        """
        if self.model_generator is None:
            raise ValueError("No models loaded")
        return next(self.model_generator)

    @DeprecationWarning
    def __get_original_model_histories(self, filter_func=lambda x: True):
        for m in filter(filter_func, self.models):
            yield m.history

    @DeprecationWarning
    def __get_pruned_model_histories(self, filter_func=lambda x: True):
        for m in filter(filter_func, self.models):
            yield m.re_history

    @DeprecationWarning
    def get_original_history(self):
        return _HistoryWrapper.new(*self.__get_original_model_histories((lambda x: x.refit_)))

    @DeprecationWarning
    def get_experiment_history(self):
        return _HistoryWrapper.new(*self.__get_pruned_model_histories(lambda x: x.refit_))


class _HistoryWrapper:
    """
    Wrapper for multiple keras.callbacks.History objects that describe "the same" process.
    e.g. in a basic lottery ticket experiment where we just want to prune 79% of the weights, there are
    2 history Wrapper objects. One holding all original training histories and one holding all pruned training histories
    """
    def __init__(self, histories: np.array):
        self.histories = histories
        if len(histories) == 0:
            self.mean = None
            self.max_ = None
            self.min_ = None
            self.stdev = None
        else:
            self.mean = np.mean(self.histories, axis=0)
            self.max_ = np.max(self.histories, axis=0)
            self.min_ = np.min(self.histories, axis=0)
            self.stdev = np.std(self.histories, axis=0)

    @classmethod
    def new(cls, *callbacks):
        """Creates a new instance from callbacks.History objects"""
        hist = np.array([h.history['val_sparse_categorical_accuracy'] for h in callbacks])
        return cls(hist)

    @classmethod
    def empty(cls, epochs):
        return cls(np.empty((0, epochs), float))

    def __add__(self, other):
        """
        Takes two history wrapper objects and appends them such, that mean max and min now go over all of the histories
        """
        histories = np.append(self.histories, other.histories, axis=0)
        return _HistoryWrapper(histories)
