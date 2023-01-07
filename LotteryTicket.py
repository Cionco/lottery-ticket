from lottery.Models import _HistoryWrapper
from lottery.Models import *
from lottery.Callbacks import FreezeCallback
from lottery.ModelSelector import *
from lottery.Combiner import *
from sklearn.metrics import accuracy_score


class LotteryTicket:
    """
    The base class for any lottery ticket experiment. The entire experiment is defined and accessed through this class.

    Attributes:
        model               --
        optimizer           --
        loss                --
        metrics             -- this and all of the above will just be passed through to the ModelIO object

        combiner            -- Defines what model combiner should be used for combination of winning tickets (Object)
        selector            -- The selector defines which models are selected to be combined. It also defines how
                                how many out of how many are selected.
        iterations          -- how often an experiment should be run to build an average on later
        pruning_percentage  -- what percentage to prune in the initial Lottery Ticket run
        name                -- the name of the folder where models and everything is stored
        io                  -- Manager object for all models used in the experiment
        pruner              -- type of pruner that should be used in the initial Lottery Ticket run
    """

    def __init__(self, model, optimizer, loss, metrics,
                 combiner=DefaultCombiner,
                 selector=AllModelsSelector(n=1),
                 iterations=1,
                 name="mnist"):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.combiner = combiner
        self.selector = selector
        self.iterations = iterations
        self.pruning_percentage = 100
        self.name = name

        self.io = None
        self.pruner = None

        self.hist_original = None
        self.hist_experiment = None

    def fit(self, x, y, validation_data, epochs, evaluate=False, test_data=None):
        """
        Runs the Lottery Ticket experiment.

        :param x:               training data
        :param y:               training labels
        :param validation_data: tuple with validation data and labels
        :param epochs:          number of epochs for training
        :param evaluate:        if true, each final model is tested and prints an accuracy score
        :param test_data:       tuple with test data and labels
        """
        self.hist_original = _HistoryWrapper.empty(epochs)
        self.hist_experiment = _HistoryWrapper.empty(epochs)

        self.io = ModelIO(self.model,
                          self.optimizer,
                          self.loss,
                          self.metrics,
                          self.pruning_percentage,
                          x=x,
                          y=y,
                          validation_data=validation_data,
                          epochs=epochs
                          )

        needed_models = self.iterations * self.selector.n
        print(needed_models)
        self.io.load_models(self.name, needed_models)

        for _ in range(self.iterations):
            self.do_lottery(evaluate, test_data, self.pruning_percentage)

    def do_lottery(self, evaluate=False, test_data=None, p=0):
        """
        Runs a single lottery ticket experiment:
            1. Take a basic fully connected trained model
            2. Prune p% of the weights
            3. Reset the remaining weights to their initial state
            4. Repeat step 1-3 for n models.
            5. Select k of the n models (combiner.selector)
            6. combine the selected models (combiner)
            7. retrain the new model
        """
        models = []  # the ModelWrappers of this iteration
        for i in range(self.selector.n):
            m = self.io.fetch()

            if evaluate:
                self.evaluate(test_data)

            m.prune(self.pruner, p)

            models.append(m)

        #  No matter if the last model is in the selected models, it will be used as the final model
        #  i.e. the experiment models are always % n.
        selected_models = self.selector.select(*models)
        for m in models:
            m.reset()
        m.source_weights = [m.model.get_weights() for m in selected_models]
        new_weights, new_maskers = self.combiner.marry(*m.source_weights)
        for m in models:
            self.hist_original += _HistoryWrapper.new(m.history)
        del models  # Free the space for the model list since it's not needed anymore.
        m.set_weights(new_weights)
        m.maskers = new_maskers

        m.refit(callbacks=[FreezeCallback(new_maskers)], **self.io.train_props)
        self.hist_experiment += _HistoryWrapper.new(m.re_history)

        if evaluate:
            self.evaluate(test_data)

    def set_pruner(self, pruner):
        self.pruner = pruner

    def set_pruning_percentage(self, p: int):
        self.pruning_percentage = p

    def evaluate(self, m, test_data):
        test_x, test_y = test_data
        y_pred = m.predict(test_x)
        print(accuracy_score(np.argmax(y_pred, axis=1), test_y))

    def plot_history(self, ylim=[0, 1]):
        """
        Plots the original training history and the experiment's history
        """
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, history in enumerate([self.hist_original, self.hist_experiment]):
            plt.plot(history.mean)
            plt.vlines(x=range(self.train_props["epochs"]), ymin=history.min_, ymax=history.max_, colors=colors[i])
        plt.legend([100, self.pruning_percentage], loc='upper left')
        plt.ylim(ylim)
        plt.show()