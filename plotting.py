from matplotlib import pyplot as plt

from chip import CDS
import pandas as pd
from model import prepare_model
from utils import get_dataframe, Parameters


def compute_signals(model, X_test):
    """
    Compute signals in X_test period
    :param X_test: a Pandas series indexed by date, with values of inputs
    :return:
    """
    # binary predict
    predictions = model.predict(X_test)

    # zip predictions to dates
    return pd.Series(data=predictions, index=X_test.index)

def sample_signals(split, name):
    def plot_signals(prices, signals, name, rp):
        """
        :param name:
        :param prices: a Series indexed by date, closing prices
        :param signals: a binary series indexed by date, 1 for up and 0 for down
        :return:
        """

        plt.plot(prices.index, prices.values, lw=0.8, label="Prices of {}".format(name))
        # You may subscript a Series directly by datetime objects
        ups = prices[signals[signals == 1].index]
        downs = prices[signals[signals == 0].index]

        # You may pass a Series itself to pyplot, it will extract values
        plt.scatter(ups.index, ups.values, c='r', s=9, label="Up signals in {} days".format(rp))
        plt.scatter(downs.index, downs.values, c='g', s=9, label="Down signals in {} days".format(rp))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show()
    df = get_dataframe(name)
    cds = CDS(df.index, df.CLOSE, df.TURN, name)
    model, X_test, y_test = prepare_model(cds, split_date=split,
                                          evaluate=True, **Parameters.standard)
    # A trader needs a trained model
    signals = compute_signals(model, X_test)

    # Get prices
    prices = cds.prices[cds.prices.index >= split]
    plt.xticks(rotation=45)
    plot_signals(prices, signals, 'IF', 60)