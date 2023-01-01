from lottery.LotteryTicket import LotteryTicket
import matplotlib.pyplot as plt


class ExperimentSearch:
    def __init__(self, architecture, optimizer, loss, metrics, combiners, selectors, pruners, iterations,
                 pruning_percentage):
        self.architecture = architecture
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.combiners = combiners
        self.selectors = selectors
        self.pruners = pruners
        self.iterations = iterations
        self.pruning_percentage = pruning_percentage

        self.results = {}

    def fit(self, train_images, train_labels, validation_data, epochs, test_data=None):

        for combiner in self.combiners:
            for selector in self.selectors:
                for pruner in self.pruners:
                    lt = LotteryTicket(self.architecture,
                                       self.optimizer,
                                       self.loss,
                                       self.metrics,
                                       combiner,
                                       selector,
                                       self.iterations)

                    lt.set_pruner(pruner)
                    lt.set_pruning_percentage(self.pruning_percentage)
                    lt.fit(train_images, train_labels, validation_data=validation_data, epochs=epochs, test_data=test_data)

                    self.results[(combiner, selector, pruner)] = lt

    def plot_results(self, ylim=[0.9, 1], loc='upper left'):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, (c, s, p) in enumerate(self.results):
            history = self.results.get((c, s, p)).io.get_experiment_history()

            plt.plot(history.mean)
            plt.vlines(x=range(len(history.mean)), ymin=history.min_, ymax=history.max_, colors=colors[i])
        plt.legend([", ".join([str(c), str(s), p.__name__]) for (c, s, p) in self.results], loc=loc)
        plt.ylim(ylim)
        plt.show()
