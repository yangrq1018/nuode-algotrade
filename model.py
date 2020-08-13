#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from chip import CDS
from utils import neighbor_percentage, kurt, k_day_return_afterward, profit_region, get_dataframe, Parameters


def get_model_cds_X_test(split, SI, param=Parameters.standard):
    """
    The days before this date are training set
    The days on and after this date are the testing set
    :param split:
    :param SI:
    :return:
    """
    df = get_dataframe(SI)
    cds = CDS(df.index, df.CLOSE, df.TURN, SI)

    # Load from pickle instead of retraining
    model, X_test, y_test = prepare_model(cds, split_date=split, load_from_disk=False, save_to_disk=False,
                                          evaluate=True, **param)

    return model, cds, X_test


def compute_Xy(cds_object, return_period, clipping_factor,
               back_price_window):
    """
    :param save_df:
    :param back_price_window:
    :param cds_object:
    :param neighbor_factor:
    :param return_period:
    :param clipping_factor:
    :return: X, y in pandas dataframe and series, already aligned along time and clipped with the same length
    """

    def _compute_back_price_window(cds_object, today, N):
        # Retrieve past prices window (None -> MAX)
        idx = cds_object.prices.index.get_loc(today)

        if N == 'max':
            return cds_object.prices[:idx + 1]
        else:
            past_prices_window = cds_object.prices[idx + 1 - N: idx + 1]
            if len(past_prices_window) < N:
                return np.array([])
            else:
                return past_prices_window

    Xd = []
    for d in cds_object.dates():
        # print('Processing date', d)
        # Remove chips with probability less than 1%, and scale up others
        # Try to reveal the major holders
        clipped_cd = cds_object.get_chip_dist(d, clip_factor=clipping_factor)
        current_price = cds_object.prices[d]
        past_prices_window = _compute_back_price_window(cds_object, d, back_price_window)

        if len(past_prices_window) < 2:
            continue

        pp_low, pp_high = min(past_prices_window), max(past_prices_window)
        # 当前价格位置
        cp_rel_pos_hist = (current_price - pp_low) / (pp_high - pp_low)

        # 平均持仓成本位置
        mc = np.sum([k * v for k, v in clipped_cd.items()])
        mc_rel_pos_hist = (mc - pp_low) / (pp_high - pp_low)
        # cp_rel_pos_hist = (current_price - mc) / mc

        # mc_rel_pos_hist = mc / cds_object.prices[d]


        # 峰度
        kurtosis = kurt(clipped_cd)
        # 当前价格筹码集中度
        # 盈利比例
        # profit_prob = profit_region(clipped_cd, current_price)

        Xd.append((d,
                   [
                       cp_rel_pos_hist,
                       mc_rel_pos_hist,
                       kurtosis,
                       # cp_neighbor_perc,
                       # profit_prob
                   ]))

    Xd = OrderedDict(Xd)
    Xd = pd.DataFrame.from_dict(Xd, orient='index')
    Xd.columns = [
        '当前价格位置',
        '平均持仓成本位置',
        '峰度',
        # '当前价格筹码集中度',
        # '盈利比例'
    ]

    # Construct return series
    yd = k_day_return_afterward(cds_object.prices, return_period)

    # Align X and y
    common_index = Xd.index.intersection(yd.index)
    Xd = Xd.loc[common_index]
    yd = yd[common_index].apply(lambda x: 1 if x > 0 else 0)

    print('X&y shape:', Xd.shape, yd.shape)
    assert Xd.shape[0] == yd.shape[0]

    return Xd, yd


def prepare_model(cds, split_date, pkl_file=None, load_from_disk=False,
                  evaluate=False, save_to_disk=False, **kwargs):
    """
    Return X_test, y_test along side the trained model
    :param cds:
    :param split_date: the date before which the market data will be used to train the model, after which the market
    data will be used to test against the strategy based on the model
    :param clipping factor
    :param back_price_window
    :return: the trained model, inputs and actual returns in the evaluation period
    """
    Xd, yd = compute_Xy(cds, **kwargs)

    X_train = Xd[Xd.index < split_date]
    X_test = Xd[Xd.index >= split_date]
    y_train = yd[yd.index < split_date]
    y_test = yd[yd.index >= split_date]

    if load_from_disk:
        print('Loading model from disk, file = {}'.format(pkl_file if pkl_file else 'sample_model.pkl'))
        model = pickle.load(open('{}.pkl'.format(pkl_file if pkl_file else 'sample_model.pkl'), 'rb'))
    else:
        # Fix random state so we will have a constant accuracy score every time in debugging
        model = RandomForestClassifier(random_state=47)
        model.fit(X_train.values, y_train.values)

    print(model)
    if save_to_disk:
        pickle.dump(model, open('pretrained/{}.pkl'.format(
            cds.name + split_date + 'RFC' + str(kwargs['return_period'])
        ), 'wb'))

    if evaluate:
        accuracy_s = accuracy_score(y_test.values, model.predict(X_test.values))
        print('----->  accuracy score:', accuracy_s)
        # Classification report
        print(classification_report(y_test, model.predict(X_test), digits=4, target_names=['跌', '涨']))
        print('实际')
        print(len(y_test[y_test == 1]), len(y_test[y_test == 0]))
        print('预测')
        pred = model.predict(X_test)
        print(len(pred[pred == 1]), len(pred[pred == 0]))
    return model, X_test, y_test

