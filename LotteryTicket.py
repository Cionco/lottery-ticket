from lottery.Models import *
from lottery.Callbacks import FreezeCallback
from sklearn.metrics import accuracy_score


class LotteryTicket:

    def __init__(self, model, optimizer, loss, metrics, combiner, iterations=1):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.combiner = combiner
        self.iterations = iterations
        self.pruning_percentage = [100]

        self.io = None
        self.pruner = None

    def fit(self, x, y, validation_data, epochs, evaluate=False, test_data=None):
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

        needed_models = len(self.pruning_percentage) * self.iterations * self.combiner.c
        print(needed_models)
        self.io.load_models("mnist", needed_models)

        for p in self.pruning_percentage:
            for _ in range(self.iterations):
                self.do_lottery(x, y, validation_data, epochs, evaluate, test_data, p)

    def do_lottery(self, x, y, validation_data, epochs, evaluate=True, test_data=None, p=100):
        weights = []  # initial pruned weights of c models
        for i in range(self.combiner.c):
            m = self.io.fetch()

            if evaluate:
                self.evaluate(test_data)

            m.prune(self.pruner, p)

            m.reset()

            weights.append(m.model.get_weights())

        m.source_weights = weights
        print("Next Step marries all weights. List of weights: ", len(weights))
        new_weights, new_maskers = self.combiner.marry(*weights)
        m.set_weights(new_weights)
        m.maskers = new_maskers

        m.refit(callbacks=[FreezeCallback(new_maskers)], verbose=0, **self.io.train_props)

        if evaluate:
            self.evaluate(test_data)

    def set_pruner(self, pruner):
        self.pruner = pruner

    def set_pruning_percentage(self, p):
        self.pruning_percentage = p

    def evaluate(self, m, test_data):
        test_x, test_y = test_data
        y_pred = m.predict(test_x)
        print(accuracy_score(np.argmax(y_pred, axis=1), test_y))
