#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

import pandas as pd

import matplotlib.pyplot as plt
from model import get_model_cds_X_test
from utils import fp


def plot_signal(split, model, X_test, cds, ax=None):
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
    # A trader needs a trained model
    signals = compute_signals(model, X_test)

    # Get prices
    prices = cds.prices[cds.prices.index >= split]
    # You may subscript a Series directly by datetime objects
    ups = prices[signals[signals == 1].index]
    downs = prices[signals[signals == 0].index]

    if not ax:
        ax = plt.subplot(111)

    # You may pass a Series itself to pyplot, it will extract values
    ax.scatter(ups.index, ups.values, c='r', s=7, alpha=0.7, label="看多信号")
    ax.scatter(downs.index, downs.values, c='g', s=7, alpha=0.7, label="看空信号")
    ax.legend(prop=fp)

# cds = CDS.from_ticker('IF')
# cds.plot_dist('2017-08-18', 0.005, aggregate=True, bin_size=10)
# cds.plot_dist('2016-06-22', 0.005, aggregate=True, bin_size=10)
# cds.plot_dist('2018-04-17', 0.005, aggregate=True, bin_size=10)
