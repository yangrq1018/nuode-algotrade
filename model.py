import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score
from utils import neighbor_percentage, kurt, k_day_return_afterward, profit_region


def compute_Xy(cds_object, neighbor_factor, return_period, clipping_factor,
               back_price_window='max', save_df=False):
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
        cp_rel_pos_hist = (cds_object.prices[d] - pp_low) / (pp_high - pp_low)
        # 平均持仓成本位置
        mc_rel = np.sum([k * v for k, v in clipped_cd.items()])
        mc_rel_pos_hist = (mc_rel - pp_low) / (pp_high - pp_low)
        # 峰度
        kurtosis = kurt(clipped_cd)
        # 当前价格筹码集中度
        cp_neighbor_perc = neighbor_percentage(current_price, neighbor_factor, clipped_cd)
        # 盈利比例
        profit_prob = profit_region(clipped_cd, current_price)

        Xd.append((d,
                   [cp_rel_pos_hist, mc_rel_pos_hist, kurtosis, cp_neighbor_perc, profit_prob]))

    Xd = OrderedDict(Xd)
    Xd = pd.DataFrame.from_dict(Xd, orient='index')
    Xd.columns = ['当前价格位置', '平均持仓成本位置', '峰度', '当前价格筹码集中度', '盈利比例']

    # Construct return series
    yd = k_day_return_afterward(cds_object.prices, return_period)

    # Align X and y
    common_index = Xd.index.intersection(yd.index)
    Xd = Xd.loc[common_index]
    yd = yd[common_index].apply(lambda x: 1 if x > 0 else 0)

    print('X&y shape:', Xd.shape, yd.shape)
    assert Xd.shape[0] == yd.shape[0]

    # Save to future use
    if save_df:
        print('Dump to disk')
        pickle.dump(Xd, open('IF_X.pkl', 'wb'))
        pickle.dump(Xd, open('IF_y.pkl', 'wb'))

    return Xd, yd


def prepare_model(cds, split_date='2018-01-01', pkl_file=None, load_from_disk=False,
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
            cds.name+split_date + 'RFC' + str(kwargs['return_period'])
        ), 'wb'))

    if evaluate:
        accuracy_s = accuracy_score(y_test.values, model.predict(X_test.values))
        print('----->  accuracy score:', accuracy_s)

    return model, X_test, y_test

# cv_book = []
#

# for F in ['IF', 'IC', 'IH']:
#     stock_index = get_dataframe(F)
#     print('$$$股指:', NAME[F])
#     for neighbor, return_period, clip_factor, back_price_window in product([0.01, 0.05, 0.1],
#                                                                        [30, 45, 60],
#                                                                        [0.01, 0.005, 0.001],
#                                                                        [30, 60, 'max']):
#         # neighbor, return_period, clip_factor, back_price_window = 0.01, 45, 0.005, 60
#         parameters = (neighbor, return_period, clip_factor, back_price_window)
#         print('------------------------')
#         print('Under parameters:', parameters)
#         cds = CDS(stock_index.index, stock_index.CLOSE, stock_index .TURN, name=F)
#         parameters = dict(neighbor_factor=0.01, return_period=45, clipping_factor=0.005, back_price_window=60)
#         model, test_X, test_y = prepare_model(cds, evaluate=True, model_type='classification', **parameters)
#         score = accuracy_score(test_y, model.predict(test_X))
#         cv_book.append([NAME[F], neighbor, return_period, clip_factor, back_price_window, score])
#
# cv_book = pd.DataFrame(cv_book, columns=['股指', '参数1', '参数2', '参数3', '参数4', '得分'])
# cv_book.to_excel('调参结果.xlsx', index=False)

#
# for SPLIT_DATE in ['2016-01-01', '2017-01-01', '2018-01-01']:
#     for C_TICKER in ['IF', 'IC', 'IH']:
#         rps = np.arange(2, 61, step=5)
#         stock_index = get_dataframe(C_TICKER)
#         cds = CDS(stock_index.index, stock_index.CLOSE, stock_index.TURN, name=C_TICKER)
#         accu_scores = []
#         for rp in rps:
#             print('rp:', rp)
#             parameters = dict(neighbor_factor=0.01, return_period=rp, clipping_factor=0.005, back_price_window=60)
#             model, X_test, y_test = prepare_model(cds, evaluate=False, split_date=SPLIT_DATE,
#                                                   model_type='classification',
#                                                   **parameters)
#             print(y_test.shape, model.predict(X_test).shape)
#             score = accuracy_score(y_test, model.predict(X_test))
#             print('Score:', score)
#             accu_scores.append(score)
#
#         plt.plot(rps, accu_scores)
#         plt.xlabel('Forecasting period')
#         plt.title(C_TICKER + ' SD:' + SPLIT_DATE)
#         plt.ylabel('Accuracy score')
#         plt.show()
