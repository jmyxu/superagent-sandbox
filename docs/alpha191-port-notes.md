# Alpha191 移植说明（GTJA 国泰君安 191 短周期价量因子）

任务：GitHub issue #46，把国泰君安 Alpha191 移植进 `scripts/alpha_factors.py`，
补全开源因子三件套（WorldQuant 101 + Qlib 158 + GTJA 191）。

## 移植基准与参考

| 角色 | 材料 | 链接 |
|---|---|---|
| **移植基准**（公式 + 实现语义） | DolphinDB 官方模块 gtja191Alpha v1.0.0（Apache-2.0） | https://github.com/dolphindb/DolphinDBModules/tree/master/gtja191Alpha |
| 公式文本对照 | BigQuant 知识库《Alpha191 因子构建公式》（与研报公开公式表一致） | https://docs.bigquant.com/wiki/doc/Pyf0TYya6H |
| 研报缺项/歧义逐条审核 | laozdao/dao-quant-research M06-06（基于 32 页研报 PDF 的逐项登记，22 条因子专属缺项、20 条语义歧义） | https://github.com/laozdao/dao-quant-research/blob/main/articles/M06-factor-validation/M06-06-gtja191-formula-reference.md |
| 独立实现对照（未引用代码） | lrb777/gtja191（pandas 移植 + 公式审核文档） | https://github.com/lrb777/gtja191 |

原研报：国泰君安证券 (2017-06)《基于短周期价量特征的多因子选股体系》
（数量化专题）。研报公式本身不满足无额外假设的严格复现——MAX(A,5) 的滚动
与逐元素含义、SMA 初值、SUMAC 窗口、Alpha166/190 的损坏表达式等均无原文
可依。本移植以 DolphinDB 模块为基准逐条落地，并与上述两处独立来源核对，
所有约定与分歧都在 `alpha_factors.py` 的 Alpha191 小节注释与模块 docstring
中声明，不把推断冒充研报原文。

## 算子语义（GTJA191 专有口径）

- `RANK(X)`：当日横截面升序分位秩，值域 (0,1]，(rank+1)/n（DolphinDB
  rowRank percent=true 同口径；本仓库用 pandas `rank(pct=True)`，并列取均值）。
- `TSRANK(X,N)`：当日 X 在最近 N 日内的 **0 起整数**升序名次（0=窗口最低，
  并列取 min）。注意不是分位——alpha117 的 `(1−TSRANK)` 依赖这一点。
- `SUM/MEAN/STD/MAX/MIN/CORR/COVIANCE(X,N)`：完整窗口（不足 N 日为 NaN）；
  STD/COV 用 ddof=1（DolphinDB mstd/mcovar 同）。
- `MAX(A,B)/MIN(A,B)`：逐元素。表 6 中形如 `MAX(A,5)` 的数值第二参数按
  **滚动 N 日**实现（DolphinDB 原版在 alpha64/91/108/119/121/127/154 等处
  误用逐元素比较）。
- `SMA(X,N,M)`：通达信式递推 y[0]=x[0]、y[t]=(M·x[t]+(N−M)·y[t−1])/N。
  DolphinDB 用 ewmMean(adjust=true)，两者暖机段有差异、收敛后一致。
- `WMA(X,N)`：权重 1..N（最新权重 N）。研报附录写"距今 i 天权重 0.9^i"，
  DolphinDB 实现为线性权重，本移植跟随基准实现。
- `DECAYLINEAR(X,N)`：权重 1..N（最新权重 N）。
- `REGBETA(Y,X,N)`：Y 对 X 的含截距滚动 OLS 斜率。alpha21/147 的
  SEQUENCE(N) 窗口按 6/12 补；alpha116 补 20。alpha149 在最近 252 日内
  筛选指数下跌日后回归（配对样本 < 2 时为 NaN）。
- `SUMAC(X,N)`：N 日滚动求和。alpha165/183 的 MAX/MIN 按**截至当日的扩展
  极值**实现（DolphinDB 对全序列取极值，回测中会引入未来信息；本仓库有
  无未来函数的硬约定），并把整个极差除以 STD。
- `alpha143 SELF`：递推初值 1.0，当日 CLOSE>DELAY(CLOSE,1) 时乘
  (CLOSE/DELAY(CLOSE,1)−1)，否则保持；停牌日输出 NaN 并保持状态。

## 数据需求

- 186 个因子：仅 OHLCV(+vwap=amount/volume)。
- gtja075/149/181/182：基准指数开/收盘（本项目以单一指数序列近似，如
  沪深 300；`compute --index-csv` 提供）。
- gtja030：Fama-French 三因子 MKT/SMB/HML（研报未给构造细节；未提供时
  该因子全 NaN，如实声明）。

## 对 DolphinDB 原版已知笔误的修正（逐条）

| 因子 | 原版问题 | 本移植 |
|---|---|---|
| alpha7 | 丢括号 A+B×C | 按公式 (A+B)×C |
| alpha28 | ×100 写进分母、第二项分母未修正 | 按笔误修正为 3×Alpha57−2×Alpha96 |
| alpha33 | 丢括号 Σ240−Σ20/220 | ((Σ240−Σ20)/220) |
| alpha81 | α 误用 1/21 | 按公式 2/21 |
| alpha104 | DELTA(CORR,5) 滞后 4 | 滞后 5 |
| alpha121/108/91/127/154/64/119/157/162/164 | 标量 min/max 逐元素 | 按公式滚动 N 日 |
| alpha146 | 丢 (A−B) 因子且分母用普通均值 | 按公式文本 MEAN(A−B,20)·(A−B)/SMA(B²,60,1) |
| alpha165/183 | 全序列极值 + 只把 MIN 除以 STD | 扩展极值 + (MAX−MIN)/STD |
| alpha166 | 分母按字面 Σ(MEAN(R,20)²,20) | 跟随（原式损坏，标注为字面解读） |
| alpha173 | 第三项误用 close | 按公式 LOG(CLOSE) 三重 SMA |
| alpha190 | 分母误乘 0/1 向量 D | 按公式 COUNT |

另一公开 pandas 移植（lrb777）在 alpha146 分母取 SMA((A−B)²,60,2)、
alpha166 按偏度指标重构等处的不同选择也已核对，本移植不跟随其推断，
全部分歧以注释形式保留在代码中。

## 验证

- `python scripts/alpha_factors.py selftest`：全部 191 个因子的形状/±inf 检查；
  截断一致性（t=199 与完整计算逐值相等）、未来扰动不变性、停牌缺口三组
  无未来函数断言覆盖 191 个因子；手工精确值覆盖 gtja002/013/014/015/020/
  080/126/143/150/171/185 及 TSRANK/SMA 递推口径；指数/三因子缺输入全
  NaN、提供后有限值。
- 真实数据：`compute --pool full_a --source tsp --max-stocks 400 --only 191
  --index-csv data/daily/000300_index.csv` 输出
  `output/factors/alpha191_latest.csv` 与 `alpha191_coverage_summary.csv`。

⚠ 历史模拟回测结果不代表未来表现，不构成投资建议。
