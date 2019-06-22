#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

import pandas as pd
from matplotlib import pyplot as plt

from chip import CDS
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


def sample_signals(split, model, X_test, cds, rp):
    # A trader needs a trained model
    signals = compute_signals(model, X_test)

    # Get prices
    prices = cds.prices[cds.prices.index >= split]
    # You may subscript a Series directly by datetime objects
    ups = prices[signals[signals == 1].index]
    downs = prices[signals[signals == 0].index]

    # You may pass a Series itself to pyplot, it will extract values
    plt.scatter(ups.index, ups.values, c='r', s=9, label="Up signals in {} days".format(rp))
    plt.scatter(downs.index, downs.values, c='g', s=9, label="Down signals in {} days".format(rp))
    plt.xlabel('Time')
    plt.ylabel('Price')
    return plt.gca()
