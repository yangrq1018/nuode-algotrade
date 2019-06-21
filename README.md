# 基于筹码分布和随机森林算法的交易策略

## 回测表现
测试于沪深300指数(399300.SZ) 数据来源：万德

### 规则
- 初始资金等分N份，以每份作为每次交易的交易额。
- 杠杆=1
- 可以做多/做空
- 手续费万分之0.23（非平今仓）
- 做空规则，设置100%准备金。
- 不设止损止盈，回测结束前清算所有仓位，累计净值以全部现金计算
```
累计净值 $1.4024823099890362
时间区间	 from 2016-06-13 to 2019-06-12, 3.00 years
累计收益率	 40.25%
年化收益率	 11.95%
标的累计收益率	 20.37%
标的年化收益率	 6.38%
超额年化收益率	 5.56%
最大回撤		 -6.80%
胜率		 80.30%
```

## Todo
- 止损
- 交易成本，中金所网站有 OK
- 概率引入 (call `predict_proba`) OK
- 收入周期阶段
- 加入做空 OK
- 绘制净资产曲线 OK 和持仓曲线
- 计算胜率OK，赔率OK，夏普比率


Remove all 50-50 probability and choose inaction


# 主要参数

- Model
    - Clip factor
    - Neighbor factor
    - Length of back price window (Relative high/low)
    - Return period
    - Train/Test partition
- Trading
    - Uncertainty tolerance
    - Initial fund partition (spending scheme)