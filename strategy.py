"""
只做多，不做空
"""
import matplotlib.pyplot as plt

class Trader:
    def __init__(self, model):
        """
        The trader needs a model to process the signals and produce trading decisions
        :param model:
        """
        self.model = model


    def compute_signals(self, X_test):
        # binary predict
        predictions = self.model.predict(X_test)


class Market:
    """
    Proxy for the market, both data and transaction handler
    """
    def __init__(self, X_data, y_data, prices):
        """
        Initialize the market with input data and return data.
        Prices can be provided to quote the real market data
        :param X_data:
        :param y_data:
        """


# todo Probability based confidence to enhance the strategy

def plot_signals(prices, signals):
    """

    :param prices: a Series indexed by date, closing prices
    :param signals: a binary series indexed by date, 1 for up and 0 for down
    :return:
    """

    plt.plot
