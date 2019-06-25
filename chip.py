#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from logger import logger
from utils import get_dataframe


class CDS:
    def __init__(self, date, price, turn, name):
        '''date, price, turn must be array like, date must an array of datetime objects
        representing the time axis, price series and turnover rate series
        '''

        self.chip_dists = []
        self.prices = {}
        self.name = name

        start = True
        for d, p, t in zip(date, price, turn):
            self.prices[d] = p
            if start:
                self.chip_dists.append((d, {p: 1}))
                start = False
            else:
                # chip distribution dictionary of yesterday
                _, last_dist = self.chip_dists[-1]
                today_dist = dict()
                for k, v in last_dist.items():
                    today_dist[k] = v * (1 - t)  # Prob staying at this state

                if p in last_dist:
                    # Update this existing price level
                    today_dist[p] = last_dist[p] + t
                else:
                    # Create a new entry
                    today_dist[p] = t

                self.chip_dists.append((d, today_dist))

        self.chip_dists = OrderedDict(self.chip_dists)
        self.prices = pd.Series(OrderedDict(sorted(self.prices.items(), key=lambda x: x[0])))

    @staticmethod
    def from_ticker(ticker):
        df = get_dataframe(ticker)
        return CDS(df.index, df.CLOSE, df.TURN, ticker)

    def get_price_on_date(self, date):
        if date not in self.prices.index:
            raise KeyError('Not a trading day in the concerning period {}'.format(date))
        return self.prices[date]

    def dates(self):
        return self.prices.index

    def get_chip_dist(self, date, clip_factor=None, aggregate=False, bin_size=None):
        '''
        date should be a string of format YYYY-MM-DD
        :param date:
        :param clip_factor: Bars with height less than max*noise_level will be filtered away
        :param aggregate: If set to True, return a chip distribution profile with bars aggregate on intervals
        :param bin_size: Control the aggregate level, the higher it is, the less bars produced
        :return: the chip distribution profile on date_str, by copying the original dictionary
        '''
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        if date not in self.chip_dists:
            # print(date, type(list(self.chip_dist.keys())[0]))
            raise ValueError('Not a trading day:', date)

        rel = self.chip_dists[date].copy()
        if clip_factor:
            logger.debug('Threshold chips')
            if clip_factor < 0 or clip_factor > 1:
                raise ValueError("Improper threshold, between 0 and 1")
            clip_factor = max(rel.values()) * clip_factor  # The filter is set to max(prob) * thresh
            for k in list(rel.keys()):
                if rel[k] < clip_factor:
                    rel.pop(k)  # Remove that price level

            # Scale up the remaining bars -> probability sums up to one
            scale_factor = 1 / sum(rel.values())
            for k, v in rel.items():
                rel[k] = v * scale_factor

        if aggregate and bin_size:
            # aggregate over the price level by ``bin_sze``
            # for index, we sum up chips in each 1 point

            rel = OrderedDict(sorted(rel.items(), key=lambda x: x[0]))

            new_rel = dict()
            for k, v in rel.items():
                box = int((k // bin_size) * bin_size)
                if box in new_rel:
                    new_rel[box] += v
                else:
                    new_rel[box] = v
            rel = new_rel
        return rel

    def plot_dist(self, date_str, **kwargs):
        """
        TODO, filter for non-significant chips at certain price levels, so that we can
        enlarge the plot and get a clear shape
        :param date_str:
        :return:
        """

        date = datetime.strptime(date_str, '%Y-%m-%d')
        dist = self.get_chip_dist(date_str, **kwargs)
        current_price = self.prices[date]
        # Partition prices as greater than and less than
        lower_dist = {k: v for k, v in dist.items() if k < current_price}
        upper_dist = {k: v for k, v in dist.items() if k >= current_price}

        profit_percentage = sum(lower_dist.values()) / (sum(upper_dist.values()) + sum(lower_dist.values()))

        plt.barh(*zip(*lower_dist.items()), height=9, color="b", label="Profit region: {}".format(
            profit_percentage
        ))
        plt.barh(*zip(*upper_dist.items()), height=9, color="y")

        # Add that day's price as cut off
        plt.axhline(y=current_price, c='k', ls='--')

        plt.xlabel('Chip')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.title('Chip distribution on {}'.format(date_str))
        plt.show()
        return plt.gca()
