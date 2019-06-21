# 基于筹码分布和随机森林算法的交易策略

## 回测表现
测试于沪深300指数(399300.SZ) 数据来源：万德

规则，只能做多不能做空，有限初始资金，固定持仓周期
```
累计净值 $1.4026796452324408
时间区间	 from 2016-06-13 to 2019-06-12, 3.04 years
累计收益率	 40.27%
年化收益率	 11.78%
标的累计收益率	 20.37%
标的年化收益率	 6.29%
超额收益率	 5.49%
```

## Todo
- 止损
- 交易成本，中金所网站有
- 概率引入 (call `predict_proba`)
- 收入周期阶段
- 加入做空
- 绘制建仓持仓图示，净资产曲线和持仓曲线


Remove all 50-50 probability and choose inaction


# 主要参数

- Model
    - Clip factor
    - Neighbor factor
    - Length of back price window (Relative high/low)
    - Return period
    - Train/Test partition
- Trading
    - Tolerance
    - Initial fund partition (spending scheme)