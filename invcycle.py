#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# from utils import fp

def get_inv_df():
    FP = r"data\库存周期数据.xlsx"
    df = pd.read_excel(FP, index_col=0)
    df.columns = ['营业收入', '产成品存货']
    return df


if __name__ == '__main__':
    from strategy import simulate_strategy
    from model import get_model_cds_X_test

    split = '2016-06-13'
    fund_partition = 50
    initial_balance = 10000
    prob_tol_factor = 0.8

    fig = plt.figure()
    ax = fig.add_subplot(211)
    # simulate_strategy(*get_model_cds_X_test(split, SI='IF'), split, fund_partition, initial_balance, prob_tol_factor,
    #                   show_plot=['price', 'signal'], ax=ax, plt_show=False)
    # plt.show()

    _, cds, _ = get_model_cds_X_test(split, SI='IF')
    ax.plot(cds.prices.index, cds.prices)

    # ax.set_xlim(left=pd.Timestamp.strptime("2016-06-13", "%Y-%m-%d"))
    # ax.xaxis.set_visible(False)
    ax2 = fig.add_subplot(212, sharex=ax)

    df = get_inv_df()
    ax2.plot(df['营业收入'].index, df['营业收入'], label="营业收入")
    ax2.plot(df['产成品存货'].index, df['产成品存货'], label="产成品存货")
    ax2.legend()
    fig.show()

    for c in ['营业收入', '产成品存货']:
        ui = cds.prices.index.intersection(df[c].index)
        s1 = cds.prices[ui]
        s2 = df[c][ui]
        print(s1.shape, s2.shape)
        print(np.corrcoef(s1, s2))

print(np.corrcoef(df['营业收入'].values, df['产成品存货'].values))
