#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.

from model import prepare_model
from chip import CDS
from utils import Parameters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

clf, X_test, y_test = prepare_model(CDS.from_ticker('IF'), '2016-06-13', **Parameters.standard)

for case in ['actual', 'predicted']:
    prediction = clf.predict(X_test)

    if case == 'actual':
        ups = X_test[y_test == 1]
        downs = X_test[y_test == 0]

    if case == 'predicted':
        ups = X_test[prediction == 1]
        downs = X_test[prediction == 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ups.values[:, 0], ups.values[:, 1], ups.values[:, 2],
               c="r", marker="o", label=f"{case} Up")
    ax.scatter(downs.values[:, 0], downs.values[:, 1], downs.values[:, 2],
               c='b', marker='^', label=f"{case} Down")

    ax.set_xlabel('current price')
    ax.set_ylabel('average cost')
    ax.set_zlabel('kurtosis')
    ax.legend(loc='best')

    fig.show()

    ax.view_init(elev=12, azim=0)
    fig.show()

    ax.view_init(elev=12, azim=90)
    fig.show()

    # fig2 = plt.figure()
