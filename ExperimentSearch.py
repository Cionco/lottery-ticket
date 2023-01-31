from lottery.LotteryTicket import LotteryTicket
import matplotlib.pyplot as plt

from lottery.Models import _HistoryWrapper


class ExperimentSearch:
    """
    Acts like a gridsearch for lottery ticket experiments. By specifying a list of combiners, selectors and pruners
    multiple combinations can be evaluated from a single object.
    """
    def __init__(self, architecture, optimizer, loss, metrics, combiners, selectors, pruners, iterations,
                 pruning_percentage, folder_name, callbacks=[]):
        self.architecture = architecture
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.combiners = combiners
        self.selectors = selectors
        self.pruners = pruners
        self.iterations = iterations
        self.pruning_percentage = pruning_percentage
        self.folder_name = folder_name

        self.results = {}
        self.original_histories = None

        self.callbacks = callbacks

    def fit(self, train_images, train_labels, validation_data, epochs, test_data=None):

        self.original_histories = _HistoryWrapper.empty(epochs)

        for combiner in self.combiners:
            for selector in self.selectors:
                for pruner in self.pruners:
                    lt = LotteryTicket(self.architecture,
                                       self.optimizer,
                                       self.loss,
                                       self.metrics,
                                       combiner,
                                       selector,
                                       self.iterations,
                                       self.folder_name,
                                       self.callbacks)

                    lt.set_pruner(pruner)
                    lt.set_pruning_percentage(self.pruning_percentage)
                    lt.fit(train_images, train_labels, validation_data=validation_data, epochs=epochs,
                           test_data=test_data)

                    self.original_histories += lt.hist_original
                    self.results[(combiner, selector, pruner)] = lt.hist_experiment

    def add_lottery_history(self, lt):
        self.results[(lt.combiner, lt.selector, lt.pruner)] = lt.hist_experiment

    def plot_results(self, ylim=[0.9, 1], loc='upper left', draw_org=True, draw_err=False, draw_std=True, draw_indices=[]):
        """
        Plots the results of the experiment. A line shows the mean, shaded area the standard deviation and error bars
        the minimum and maximum value. By default, all histories are shown with their standard deviation
        :param ylim:            the part of the y axis that should be plotted
        :param loc:             location of the legend, directly passed into the pyplot.legend method
        :param draw_org:        plot the initial training history of the networks
        :param draw_err:        plot maximum and minimum errorbars
        :param draw_std:        plot standard deviation
        :param draw_indices:    list of indices of all entries in the results dictionary that should be plotted.
                                Permuation can be manipulated by reordering this list. An empty list equals
                                    list(range(len(self.results)))
        """
        if len(draw_indices) == 0:
            draw_indices = list(range(len(self.results)))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        if draw_org:
            plt.plot(self.original_histories.mean)
            if draw_err:
                plt.vlines(x=range(len(self.original_histories.mean)),
                           ymin=self.original_histories.min_,
                           ymax=self.original_histories.max_)
            if draw_std:
                plt.fill_between(x=range(len(self.original_histories.mean)),
                                 y1=self.original_histories.mean + self.original_histories.stdev,
                                 y2=self.original_histories.mean - self.original_histories.stdev,
                                 alpha=0.5)

        drawing_index = -1
        for i in draw_indices:
            key = list(self.results.keys())[i]
            history = self.results.get(key)

            drawing_index += 1

            plt.plot(history.mean)

            if draw_err:
                plt.vlines(x=range(len(history.mean)), ymin=history.min_, ymax=history.max_,
                           colors=colors[drawing_index + (1 if draw_org else 0)])
            if draw_std:
                plt.fill_between(x=range(len(history.mean)), y1=history.mean + history.stdev,
                                 y2=history.mean - history.stdev,
                                 facecolor=colors[drawing_index + (1 if draw_org else 0)], alpha=0.5)

        def get_legend_entry(key):
            return ", ".join([str(key[0]), str(key[1]), key[2].__name__])
        plt.legend((["Original"] if draw_org else []) + [get_legend_entry(list(self.results.keys())[i])
                                                         for i in draw_indices]
                   , loc=loc, bbox_to_anchor=(1.04, 1))
        plt.ylim(ylim)
        plt.show()