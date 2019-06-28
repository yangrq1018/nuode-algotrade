#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

import pandas as pd
import os
from matplotlib.font_manager import FontProperties
from matplotlib import style


DATA_DIR = "data"
# style.use('seaborn')
fp = FontProperties('SimHei', size=12)  # 雅黑字体


def get_dataframe(index):
    """
    For IF (沪深300期货) as example, 2005-4-8 is the first day when both turn and close price data
    is available.
    :param index:
    :return:
    """
    TICKER = dict(IF='399300.SZ', IC='399905.SZ', IH='000016.SH')
    URL = DATA_DIR + os.sep + TICKER[index] + '.csv'
    df = pd.read_csv(URL, index_col=0, parse_dates=[0], engine='python').drop('DEALNUM', axis=1).dropna(
        subset=['CLOSE', 'TURN'])
    df.TURN = df.TURN / 100  # Percentage turnover rate
    print('Get dataframe for {}, shape={}'.format(index, df.shape))
    return df


def neighbor_percentage(cp, factor, distribution):
    prob_sum = 0
    for k, v in distribution.items():
        if cp * (1 - factor) < k < cp * (1 + factor):
            prob_sum += v
    return prob_sum


def kurt(distribution):
    mean = sum([k * v for k, v in distribution.items()])
    m4 = sum([probability * (price - mean) ** 4 for price, probability in distribution.items()])
    m2 = sum([probability * (price - mean) ** 2 for price, probability in distribution.items()])
    return m4 / (m2 ** 2) - 3


def k_day_return_afterward(prices: pd.Series, K):
    """
    :param prices: a pandas series
    :param K:
    :return: the return series, with leftmost date of the window as labels
    """

    return prices.rolling(K).apply(lambda x: (x[-1] - x[0]) / x[0]).shift(-(K - 1)).dropna()


def profit_region(cd, current_price):
    prob = 0
    for k, v in cd.items():
        if k > current_price:
            prob += v
    return prob


class Parameters:
    standard = dict(return_period=60, clipping_factor=0.005, back_price_window=60)


FUTURES = ['IF', 'IC', 'IH']
CH_NAME = dict(IF='沪深300', IC='中证500', IH='上证50')
TICKER = dict(IF='399300.SZ', IC='399905.SZ', IH='000016.SH')


class TradingPeriodEnds(Exception):
    pass


def annualize(rt, n_years):
    return (1 + rt) ** (1 / n_years) - 1
