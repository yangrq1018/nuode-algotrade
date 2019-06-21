"""
规则只做多，不做空，M等分初始资金，RP天后平仓，只有平仓后可以继续买入
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emoji import emojize
from sklearn.ensemble import RandomForestClassifier

from model import get_model_cds_X_test
from utils import TradingPeriodEnds, annualize

RP = 60

plt.style.use('seaborn')


class Trader:
    def __init__(self, trained_model: RandomForestClassifier, time_axis, tolerance):
        """
        The trader needs a model to process the signals and produce trading decisions
        :param trained_model:
        :param fund_partition: split initial_balance in X equal slices, each long position is one slice
        """
        self.model = trained_model
        self.time_axis = time_axis
        self._broker = None
        self._tolerance = tolerance

    def register_broker(self, broker):
        self._broker = broker

    def decide(self, inputs):
        """
        Decide if long one unit of index in today.
        Forbid purchasing in the last 60 days, we must liquidate all open positions at the end.
        As return after June 12 is not known.
        last RP day, close existing position only.
        :param prob_regulator: an integer between 0 and 1, the lower it is, the stronger confidence we need
        to open a position
        :param today: a Timestamp
        :param inputs: the indicators for today
        :return: -1 for inaction, 1 for long signal, 0 for short signal
        """

        if len(inputs) == 0:
            return -1
        prob_pred = self.model.predict_proba(inputs.values.reshape((1, -1)))[0]
        if (prob_pred[0] * prob_pred[1]) >= (0.25 * self._tolerance):  # Indifference
            return -1
        decision = self.model.predict(inputs.values.reshape((1, -1)))[0]
        return decision

    def close_long(self, today):
        """
        做多平仓
        :param today:
        :return: the number of shares to sell (close long position)
        """
        # Look up leger, if a holding is longed 60 trading days ago, sell it
        # use Index.get_loc to get position of an element

        # 我们在Day 0的收盘价买入，在Day 59日的收盘价卖出，这是一个60天的窗口，回报时间段为60天，在数据帧中是60行
        before_day_index = self.time_axis.get_loc(today) - (RP - 1)

        if before_day_index < 0:
            return 0

        return self._broker.leger['long'][self.time_axis[before_day_index]]

    def close_short(self, today):
        """
        做空平仓
        :param today:
        :return:
        """
        before_day_index = self.time_axis.get_loc(today) - (RP - 1)

        if before_day_index < 0:
            return 0

        return self._broker.leger['short'][self.time_axis[before_day_index]]


class Market:
    def __init__(self, chip_info, prices):
        """
        The market has knowledge about the chips
        :param chip_info: a Pandas dataframe containing five X variables
        """
        self._chip_info = chip_info
        self._prices = prices

    def get_chip_stats(self, day):
        """
        :param day: Timestamp
        :return:
        """
        try:
            return self._chip_info.loc[day, :]
        except KeyError:
            return []

    def quote_price(self, day):
        """
        Get index closing price on day
        :param day:
        :return:
        """
        return self._prices[day]


class Clock:
    def __init__(self, time_axis):
        now = 0
        self._time_axis = time_axis

    def forward(self):
        self.now += 1

    def today(self):
        return self._time_axis[self.now]


class Broker:
    """
    中金所 2019年4月公告，交易手续费为成交金额的万分之零点二三，平今仓手续费为成交金额的万分之三点四五
    """

    def __init__(self, time_axis, init_balance, partition=10):
        """
        For short positions, cover after 60 days. On date 59, draw proceed, buy back share. Return deposit to cash
        pool, any gain or loss is credit/debit to cash pool.
        :param time_axis: a pandas DatetimeIndex, should cover up to 2019/6/12, the trading period
        :param init_balance:
        """

        self.commission_rate = 0.23 / 10000

        self._time_axis = time_axis
        self.cash_bal = init_balance
        self.short_proceed = 0  # short share proceed account
        self.deposit = 0  # Short deposit account
        self.open_close_count = 0 # Open/Close transaction

        self.holdings = 0
        self.obligations = 0

        self.investment_cash_unit = init_balance / partition
        self._clock = 0
        self._trader = None
        self._market = None
        self._trader = None

        # Leger object -> keep track of transactions and positions
        # use numpy float type, integer 0 will only hold integer values only
        self.leger = dict()
        self.leger['long'] = pd.Series(data=np.float64(0), index=self._time_axis)
        self.leger['sell'] = pd.Series(data=np.float64(0), index=self._time_axis)
        self.leger['long_total'] = np.float64(0)
        self.leger['short_total'] = np.float64(0)
        self.leger['short'] = pd.Series(data=np.float64(0), index=self._time_axis)
        self.leger['cover'] = pd.Series(data=np.float64(0), index=self._time_axis)
        self.leger['net_worth'] = pd.Series(data=np.float64(0), index=self._time_axis)
        self.leger['win_count'] = np.int64(0)
        self.leger['gain_total'] = np.float64(0)
        self.leger['loss_count'] = np.int64(0)
        self.leger['loss_total'] = np.float64(0)

    def register_trader_and_market(self, t: Trader, m: Market):
        self._trader = t
        self._market = m

    def today(self):
        return self._time_axis[self._clock]

    def net_worth(self):
        """
        Compute net worth profile
        :return:
        """

        cp = self._market.quote_price(self.today())
        return self.cash_bal + self.deposit + self.short_proceed + \
               self.holdings * cp - self.obligations * cp

    def proceed(self):
        """
        todo, use a centralized clock
        Move forward one day
        :return:
        """
        today = self.today()
        current_price = self._market.quote_price(today)

        print('------Trading day {}--------'.format(self.today().strftime('%Y-%m-%d')))

        # Close position
        sell = self._trader.close_long(today)
        if sell:
            print('[Sell] {:.2f} units at ${}'.format(sell, self._market.quote_price(today)))
            notional_amt = self._market.quote_price(today) * sell
            self.cash_bal += notional_amt
            self.holdings -= sell
            self.leger['sell'][today] = sell

            commission = notional_amt * self.commission_rate
            self.cash_bal -= commission
            print('[Commission charge] ${:.8f}'.format(commission))

            # Determine win or loss
            if notional_amt >= self.investment_cash_unit:
                self.leger['win_count'] += 1
                self.leger['gain_total'] += notional_amt - self.investment_cash_unit
            else:
                self.leger['loss_count'] += 1
                self.leger['loss_total'] += notional_amt - self.investment_cash_unit
            self.open_close_count += 1


        cover = self._trader.close_short(today)
        if cover:
            print('[Cover] {:.2f} units at ${}'.format(cover, self._market.quote_price(today)))
            # Return money from deposit and short_proceed
            self.deposit -= self.investment_cash_unit
            self.short_proceed -= self.investment_cash_unit
            self.cash_bal += self.investment_cash_unit * 2

            # Buy back stock
            notional_amt = self._market.quote_price(today) * cover
            if self.cash_bal < notional_amt:
                raise ValueError('Insufficient cash to cover short position')
            self.cash_bal -= notional_amt
            self.obligations -= cover
            commission = notional_amt * self.commission_rate
            self.cash_bal -= commission
            print('[Commission charge] ${:.8f}'.format(commission))

            if notional_amt <= self.investment_cash_unit:
                self.leger['win_count'] += 1
                self.leger['gain_total'] += notional_amt - self.investment_cash_unit
            else:
                self.leger['loss_count'] += 1
                self.leger['loss_total'] += notional_amt - self.investment_cash_unit
            self.open_close_count += 1


        # New position
        judgement = self._trader.decide(self._market.get_chip_stats(today))
        if judgement == -1:
            print('Indifference -> inaction')
        elif judgement == 1:  # Long signal
            # Validate balance
            if self.cash_bal >= self.investment_cash_unit:
                # buy
                # can only invest cash = investment_cash_unit
                long_units = self.investment_cash_unit / current_price

                print('[Long] {:.2f} units of index at ${}'.format(
                    long_units, current_price
                ))

                self.cash_bal -= self.investment_cash_unit
                self.holdings += self.investment_cash_unit / current_price
                self.leger['long'][today] = long_units

                # Commission charge
                commission = self.investment_cash_unit * self.commission_rate
                self.cash_bal -= commission
                print('[Commission charge] ${:.8f}'.format(commission))
            else:
                print('[Insufficient fund] on {}, deny long order'.format(self.today()))

        elif judgement == 0:  # Short signal
            if self.cash_bal >= self.investment_cash_unit:
                # short
                short_units = self.investment_cash_unit / current_price
                print('[Short] {:.2f} units of index at ${}'.format(short_units, current_price))

                # freeze deposit and short_proceed, add obligation
                self.cash_bal -= self.investment_cash_unit
                self.deposit += self.investment_cash_unit
                self.obligations += short_units
                self.short_proceed += self.investment_cash_unit

                self.leger['short'][today] = short_units

                commission = self.investment_cash_unit * self.commission_rate
                self.cash_bal -= commission
                print('[Commission charge] ${:.8f}'.format(commission))
            else:
                print('[Insufficient fund] on {}, deny short order'.format(self.today()))

        # record net worth on leger
        self.leger['net_worth'][today] = self.net_worth()

        self._clock += 1
        if self._clock == len(self._time_axis):
            # if self.holdings != 0:
            # raise ValueError('we should have sold out all holdings, remain holdings={}'.format(self.holdings))
            raise TradingPeriodEnds('Trading period ends')


def mdd(net_worths):
    """
    Compute maximum draw down (MDD)
    :param net_worths:
    :return:
    """

    def dd(s): return (np.min(s) - s[0]) / s[0]

    return np.min([dd(net_worths[s:]) for s in net_worths.index])


def evaluate(broker, cds, split, initial_balance, show=[]):
    last_trading_day = pd.Timestamp.strptime('2019-06-12', '%Y-%m-%d')
    first_trading_day = pd.Timestamp.strptime(split, '%Y-%m-%d')

    print(emojize('>Back testing results', use_aliases=True))

    print('剩余现金 ${:.2f}'.format(broker.cash_bal))
    print('剩余抵押金 ${:.2f}'.format(broker.deposit))
    print('剩余保管金 ${:.2f}'.format(broker.short_proceed))
    print('剩余股票 {:.6f} 股'.format(broker.holdings))
    print('剩余责任 {:.6f} 股'.format(broker.obligations))

    print('累计净值 ${}'.format(broker.cash_bal / initial_balance))

    ac_return = broker.cash_bal / initial_balance - 1
    t_days = (last_trading_day - first_trading_day).days
    years = t_days / 365
    an_return = annualize(ac_return, years)
    bm_ac_return = cds.prices[last_trading_day] / cds.prices[first_trading_day] - 1
    bm_an_return = annualize(bm_ac_return, years)
    excess_return = an_return - bm_an_return
    max_drawdown = mdd(broker.leger['net_worth'])
    win_ratio = broker.leger['win_count'] / (broker.leger['win_count'] + broker.leger['loss_count'])
    gain_loss_ratio = (broker.leger['gain_total']/broker.leger['win_count']) /\
                      (-(broker.leger['loss_total']/broker.leger['loss_count']))

    print('时间区间\t from {} to {}, {:.2f} years'.format(
        first_trading_day.strftime('%Y-%m-%d'),
        last_trading_day.strftime('%Y-%m-%d'),
        years
    ))
    print('累计收益率\t {:.2f}%'.format(ac_return * 100))
    print('年化收益率\t {:.2f}%'.format(an_return * 100))
    print('标的累计收益率\t {:.2f}%'.format(bm_ac_return * 100))
    print('标的年化收益率\t {:.2f}%'.format(bm_an_return * 100))
    print('超额年化收益率\t {:.2f}%'.format(excess_return * 100))
    print('最大回撤\t\t {:.2f}%'.format(max_drawdown * 100))
    print('胜率\t\t {:.2f}%'.format(win_ratio * 100))
    print('赔率\t\t {:.2f}%'.format(gain_loss_ratio * 100))
    print('{} = {}(win) + {}(loss)'.format(broker.open_close_count, broker.leger['win_count'], broker.leger['loss_count']))

    if 'net_worth_curve' in show:
        broker.leger['net_worth'].plot()
        plt.title('Net worth curve')
        plt.show()

    return excess_return, max_drawdown


def simulate_strategy(model, cds, X_test, split, fund_partition, initial_balance, tolerance, show=[]):
    trading_period = cds.prices.index[cds.prices.index >= split]

    # Instantiate trader, market and broker
    trader = Trader(model, trading_period, tolerance=tolerance)
    market = Market(X_test, cds.prices)
    broker = Broker(trading_period, init_balance=initial_balance, partition=fund_partition)
    broker.register_trader_and_market(trader, market)
    trader.register_broker(broker)

    while True:
        try:
            broker.proceed()
        except TradingPeriodEnds as e:
            print(e)
            return evaluate(broker, cds, split, initial_balance, show=show)


def er_mdd():
    split = '2016-06-13'
    fund_partition = 50
    initial_balance = 1000

    # simulate_strategy(*get_model_cds_X_test(split), split, fund_partition, initial_balance, 0.97, show=['net_worth_curve'])

    fig, ax1 = plt.subplots()

    env = get_model_cds_X_test(split)

    x = np.arange(0.6, 1.01, 0.01)

    ers, mdds = zip(*[simulate_strategy(*env, split, fund_partition, initial_balance, tol)
                      for tol in x])

    ax1.plot(x, ers, c='navy', label="Excess return")
    ax1.set_ylabel('Excess return')
    ax2 = ax1.twinx()
    ax2.plot(x, mdds, c='tomato', label="Max drawdown")
    ax2.set_ylabel('Max Drawdown')
    plt.xlabel('Probability control')
    fig.legend()
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    split = '2016-06-13'
    fund_partition = 50
    initial_balance = 1000

    simulate_strategy(*get_model_cds_X_test(split), split, fund_partition, initial_balance, 0.8,
                      show=[])

    # er_mdd()
    # sample_signals('2016-06-13', 'IF')
