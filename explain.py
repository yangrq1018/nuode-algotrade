#  Copyright (c) 2019. All rights reserved.
#  Author: Ruoqi Yang
#  @Imperial College London, HKU alumni
#  mailto: yangrq@connect.hku.hk
#  This file is part of the quantitative research of Nuode Fund, contact
#  service@nuodefund.com for commercial use.
from chip import CDS
from model import prepare_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import fp, Parameters



clf, X_test, y_test = prepare_model(CDS.from_ticker('IF'), '2016-06-13', **Parameters.standard)

for case in ['实际', '预测']:
    prediction = clf.predict(X_test)

    if case == '实际':
        ups = X_test[y_test == 1]
        downs = X_test[y_test == 0]

    if case == '预测':
        ups = X_test[prediction == 1]
        downs = X_test[prediction == 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ups.values[:, 0], ups.values[:, 1], ups.values[:, 2],
               c="r", marker="o", label="{}上升".format(case))
    ax.scatter(downs.values[:, 0], downs.values[:, 1], downs.values[:, 2],
               c='b', marker='^', label="{}下降".format(case))

    ax.set_xlabel('当前价格位置', fontproperties=fp)
    ax.set_ylabel('平均持仓成本位置', fontproperties=fp)
    ax.set_zlabel('超额峰度', fontproperties=fp)
    ax.legend(loc='best', prop=fp)

    fig.show()

    ax.view_init(elev=12, azim=0)
    fig.show()

    ax.view_init(elev=12, azim=90)
    fig.show()

    fig2 = plt.figure()
