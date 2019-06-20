"""
规则只做多，不做空，M等分初始资金，RP天后平仓，只有平仓后可以继续买入
"""
from collections import OrderedDict
from emoji import emojize

import matplotlib.pyplot as plt
import pandas as pd

from chip import CDS
from model import prepare_model
from utils import get_dataframe, Parameters, TradingPeriodEnds, annualize

RP = 60


class Trader:
    def __init__(self, trained_model, time_axis):
        """
        The trader needs a model to process the signals and produce trading decisions
        :param trained_model:
        :param fund_partition: split initial_balance in X equal slices, each long position is one slice
        """
        self.model = trained_model
        self.time_axis = time_axis
        self._broker = None

    def register_broker(self, broker):
        self._broker = broker

    def compute_signals(self, X_test):
        """
        Compute signals in X_test period
        :param X_test: a Pandas series indexed by date, with values of inputs
        :return:
        """
        # binary predict
        predictions = self.model.predict(X_test)

        # zip predictions to dates
        return pd.Series(data=predictions, index=X_test.index)

    def decide_long_action(self, inputs):
        """
        todo Neutral label for return == 0 ?
        Decide if long one unit of index in today
        :param today: a Timestamp
        :param inputs: the indicators for today
        :return:
        """

        if len(inputs) == 0:
            # Forbid purchasing in the last 60 days, we must liquidate all assets at the end
            # As return after June 12 is not known, the return are not settled
            return False
        decision = self.model.predict(inputs.values.reshape((1, -1)))[0]
        assert decision == 0 or decision == 1
        return decision

    def decide_short_action(self, today):
        # Look up leger, if a holding is longed 60 trading days ago, sell it
        # use Index.get_loc to get position of an element

        # 我们在Day 0的收盘价买入，在Day 59日的收盘价卖出，这是一个60天的窗口，回报时间段为60天，在数据帧中是60行
        before_day = self.time_axis[self.time_axis.get_loc(today) - (RP - 1)]

        if self._broker.leger[before_day]['long']:
            # Sell that many unit of index that we bought 59 days ago
            return self._broker.leger[before_day]['long']
        else:
            return False


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
    def __init__(self, time_axis, init_balance, partition=10):
        """

        :param time_axis: a pandas DatetimeIndex, should cover up to 2019/6/12, the trading period
        :param init_balance:
        """
        self._time_axis = time_axis
        self.cash_bal = init_balance
        self.holdings = 0
        self.investment_cash_unit = init_balance / partition
        self._clock = 0
        self._trader = None
        self._market = None
        self._trader = None

        self.leger = OrderedDict([(k, dict(long=None, short=None)) for k in self._time_axis])

    def register_trader_and_market(self, t: Trader, m: Market):
        self._trader = t
        self._market = m

    def today(self):
        return self._time_axis[self._clock]

    def proceed(self):
        """
        todo, use a centralized clock
        Move forward one day
        :return:
        """
        today = self.today()
        # Ask trader
        print('-------On trading day {}----------'.format(self.today().strftime('%Y-%m-%d')))

        # Decide short
        short = self._trader.decide_short_action(today)

        if short:
            print('[Sell] {:.2f} units at ${}'.format(short, self._market.quote_price(today)))
            self.cash_bal += self._market.quote_price(today) * short
            self.holdings -= short
            self.leger[today]['short'] = short

        buy = self._trader.decide_long_action(self._market.get_chip_stats(today))
        if buy:
            # Validate balance
            if self.cash_bal >= self.investment_cash_unit:
                # buy
                current_price = self._market.quote_price(today)
                # can only invest cash = investment_cash_unit
                long_units = self.investment_cash_unit / self._market.quote_price(today)

                print('[Long] {:.2f} units of index at ${}'.format(
                    long_units, current_price
                ))

                self.cash_bal -= self.investment_cash_unit
                self.holdings += self.investment_cash_unit / current_price
                self.leger[today]['long'] = long_units
            else:
                print('[Insufficient fund] on {}, deny long order'.format(self.today()))

        self._clock += 1

        if self._clock == len(self._time_axis):
            # if self.holdings != 0:
            # raise ValueError('we should have sold out all holdings, remain holdings={}'.format(self.holdings))
            raise TradingPeriodEnds('Trading period ends')


def evaluate_strategy():
    # The days before this date are training set
    # The days on and after this date are the testing set
    split = '2016-06-13'
    last_trading_day = pd.Timestamp.strptime('2019-06-12', '%Y-%m-%d')
    first_trading_day = pd.Timestamp.strptime(split, '%Y-%m-%d')
    fund_partition = 50
    initial_balance = 1000

    df = get_dataframe('IF')
    cds = CDS(df.index, df.CLOSE, df.TURN, 'IF')

    # Load from pickle instead of retraining
    model, X_test, y_test = prepare_model(cds, split_date=split, load_from_disk=False, save_to_disk=True,
                                          evaluate=True, **Parameters.standard)

    trading_period = cds.prices.index[cds.prices.index >= split]
    # Instantiate trader, market and broker

    trader = Trader(model, trading_period)
    market = Market(X_test, cds.prices)
    broker = Broker(trading_period, init_balance=initial_balance, partition=fund_partition)

    broker.register_trader_and_market(trader, market)
    trader.register_broker(broker)

    while True:
        try:
            broker.proceed()
        except TradingPeriodEnds as e:
            print(e)
            print(emojize('>:chart_with_upwards_trend::moneybag::dollar: Back testing results', use_aliases=True))
            print('剩余现金 ${:.2f}'.format(broker.cash_bal))
            print('剩余股票 {:.6f} shares'.format(broker.holdings))
            print('累计净值 ${}'.format(broker.cash_bal / initial_balance))

            ac_return = broker.cash_bal / initial_balance - 1
            t_days = (last_trading_day - first_trading_day).days
            years = t_days / 360
            an_return = annualize(ac_return, years)
            bm_ac_return = cds.prices[last_trading_day] / cds.prices[first_trading_day] - 1
            bm_an_return = annualize(bm_ac_return, years)
            excess_return = an_return - bm_an_return


            print('时间区间\t from {} to {}, {:.2f} years'.format(
                first_trading_day.strftime('%Y-%m-%d'),
                last_trading_day.strftime('%Y-%m-%d'),
                years
            ))
            print('累计收益率\t {:.2f}%'.format(ac_return * 100))
            print('年化收益率\t {:.2f}%'.format(an_return * 100))
            print('标的累计收益率\t {:.2f}%'.format(bm_ac_return * 100))
            print('标的年化收益率\t {:.2f}%'.format(bm_an_return * 100))
            print('超额收益率\t {:.2f}%'.format(excess_return * 100))
            break



# todo Probability based confidence to enhance the strategy


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


def sample_signals(split, name):
    df = get_dataframe(name)
    cds = CDS(df.index, df.CLOSE, df.TURN, name)
    model, X_test, y_test = prepare_model(cds, split_date=split, model_type='classification',
                                          evaluate=True, **Parameters.standard)
    # A trader needs a trained model
    trader = Trader(trained_model=model, time_axis=None)
    signals = trader.compute_signals(X_test)

    # Get prices
    prices = cds.prices[cds.prices.index >= split]
    plt.xticks(rotation=45)
    plot_signals(prices, signals, 'IF', 60)


if __name__ == '__main__':
    evaluate_strategy()