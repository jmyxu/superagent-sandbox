# 经典选股因子调研（量化学习项目第 6 步 · 横截面选股）

> 本调研为第 6 步 scripts/screener.py 的因子选型依据，全部结论均来自
> 公开学术/行业资料（Sources 节注明出处）。本步回测只落地**动量**与
> **波动率（低波动）**两个因子；价值因子的 A 股证据在此记录，实现
> 留作后续扩展。
>
> ⚠ 本调研与后续回测均为历史模拟，仅用于学习方法验证；历史模拟回测
> 结果不代表未来表现，不构成投资建议。

---
# Classic Quantitative Stock-Selection Factors for Chinese A-Shares (CSI 300 / CSI 500)

This note summarizes the classic cross-sectional factors most relevant to an educational factor project on the CSI 300 (large caps) / CSI 500 (mid caps) universe, with A-share-specific evidence and implementation caveats. It assumes a data source such as CSMAR or Wind; it is research-only, no code.

## 1. Momentum

**Definition.** Jegadeesh & Titman (1993) rank stocks on cumulative return over a formation period J (typically 6-12 months) skipping the most recent month — the "12-1" return — and hold for K months; winner-minus-loser deciles earned roughly 1% per month in the US. Common variants: 6/6, 12/1, 12/3, with overlapping portfolios to smooth turnover. Implementations range from raw cumulative-return ranking to **residual momentum** (Blitz, Huij & Martens 2011), which regresses stock returns on the Fama-French factors and applies momentum to the residuals; residual momentum is more stable and less crash-prone than raw momentum.

**A-share behavior.** The key China-specific fact: classic medium-horizon momentum is weak or absent in A-shares, while **short-term reversal (1 week - 1 month) is strong**. A review of the Chinese momentum literature concludes monthly J/K momentum fails; Cakici, Chan & Topyan (2017) find reversal among the significant anomalies in China; Cheema & Nartea show Chinese momentum is state-dependent, appearing mainly in up markets; Hsu et al. (2018) similarly report weak momentum but strong reversal. Retail-dominated turnover and speculative trading drive reversal. For a CSI 300/500 factor library: implement 1-month (or 5-day) reversal as the "momentum-family" factor, or residual/idiosyncratic momentum with a skipped month; treat classic 12-1 momentum as a known-weak baseline.

**Pitfalls.** Momentum crashes in rebounds after bear markets (Daniel & Moskowitz 2016); high turnover and costs; China's T+1 rule and price limits make fast reversal strategies harder to execute.

## 2. Low Volatility

**Anomaly.** Low-risk stocks earn higher risk-adjusted returns than CAPM predicts. Definitions: **realized volatility** (standard deviation of daily returns over 60-250 days); **idiosyncratic volatility (IVOL)** — Ang, Hodrick, Xing & Zhang (2006) find high IVOL predicts low returns, and Ang et al. (2009) confirm this internationally, including China; **CAPM beta**, traded by Frazzini & Pedersen's (2014) Betting-Against-Beta. Proposed explanations: leverage constraints and institutional benchmark/lottery preferences (Baker, Bradley & Wurgler 2011).

**Defensiveness.** Low-vol portfolios tend to lag in speculative rallies and outperform in stress periods, giving the factor a defensive character; sector concentration (banks, utilities) is a common unintended bet.

**A-share evidence.** The anomaly replicates in China: Hsu et al. (2018) document low-vol/low-beta premia in A-shares; practitioner studies (e.g., Acadian's "Risk Mispricing in China A-Shares") and CSMAR-based academic theses find robust low-vol spreads, often amplified by retail lottery demand for high-volatility stocks. IVOL and MAX effects are pronounced in the retail-dominated A-share market. Implementation: 60-120-day realized vol, or IVOL versus the CH-3/CH-4 model; neutralize industries.

## 3. Value

**Classic definitions.** Book-to-price (B/P), earnings-to-price (E/P), cash-flow-to-price; Fama-French's HML (1993) buys high-B/M and shorts low-B/M stocks. Value is the workhorse factor: cheap stocks outperform expensive ones on average, at the cost of occasional deep drawdowns (value traps).

**A-share applicability.** Liu, Stambaugh & Yuan (2019) is the key reference: in China **E/P works better than B/P**, and B/P is contaminated by the shell-value premium of the smallest listed firms; they exclude the bottom 30% by market cap and build their CH-3/CH-4 factor models around EP. Hu, Chen, Shao & Wang (2019) construct Chinese size and value factors with adjustments for ST stocks, IPO effects and trading rules. The value premium in China exists but is small-cap-driven and sensitive to construction choices; within CSI 300 (large caps) value is milder, while CSI 500 (mid caps) shows more spread.

**Pitfalls.** Value traps (state-owned, distressed or fraud-prone firms), weak accounting quality, heavy sector tilts (banks, steel are persistently cheap), and shell-value contamination at the bottom of the size distribution.

## 4. Size and Quality (complementary)

**Size.** Small-cap premia have historically been large in China but are entangled with shell values and liquidity; the Liu-Stambaugh-Yuan 30% exclusion is the standard remedy. Within CSI 300/CSI 500 the extreme microcaps are already excluded, so the size spread is muted relative to the full market.

**Quality.** Gross profitability (Novy-Marx 2013) and the Quality-Minus-Junk composite — profitability, growth, safety, payout (Asness, Frazzini & Pedersen 2019). Quality tilts help avoid value traps and combine well with value and low-vol; profitability-based quality measures are the most replicated in A-shares.

## 5. Factor Construction Conventions

- **Universe hygiene:** point-in-time fundamentals; drop ST/*ST and suspended stocks, and IPOs younger than 6-12 months; keep delisted stocks in the history to avoid survivorship bias.
- **Cross-sectional standardization:** at each rebalance, convert raw factor values to z-scores or (preferred for skewed factors) percentile ranks within the universe.
- **Winsorization:** clip extremes at 1%/99% or 3xMAD before standardizing.
- **Neutralization:** regress each factor on industry dummies and log market cap and keep the residual (standard in WorldQuant-style pipelines; Kakushadze 2016); this removes unintended sector/size bets.
- **Aggregation:** combine factors with equal- or IC-weighted z-scores; monitor cross-factor correlations and turnover.

## 6. Backtest Design Notes

- **Rebalancing:** monthly or quarterly; skip one month between formation and holding for momentum/reversal to dodge the bid-ask bounce; compute signals on rebalance-date closes and trade the next day's open/VWAP (no look-ahead).
- **Transaction costs:** commission about 0.02-0.03% per side (plus a CNY 5 minimum), stamp duty 0.05% on sells (halved in Aug 2023 from 0.1%), transfer fees; model price-limit (10%/20%) slippage and suspension risk.
- **Shorting is severely restricted in A-shares:** treat long-short results as hypothetical; prefer long-only top-quintile or benchmark-relative portfolios, or hedge with index futures.
- **Survivorship:** use full historical constituent data including delisted names (CSMAR/Wind provide these); CSI 300/500 index backfill histories embed survivorship bias.
- **Evaluation:** IC/ICIR, quintile spread returns, turnover and drawdowns — all net of costs; pre-register factor definitions to limit data snooping.

## Sources

1. Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency", Journal of Finance — [onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x)
2. "Momentum effects in China: A review of the literature and an empirical explanation of prevailing controversies", Research in International Business and Finance 47 (2019) — [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0275531918302046)
3. Blitz, Huij & Martens (2011), "Residual momentum", Journal of Empirical Finance 18(3) — [doi.org](https://doi.org/10.1016/j.jempfin.2011.01.003)
4. Daniel & Moskowitz (2016), "Momentum crashes", Journal of Financial Economics — [nber.org](https://www.nber.org/papers/w20439)
5. Cakici, Chan & Topyan (2017), "Cross-sectional stock return predictability in China", European Journal of Finance — [tandfonline.com](https://www.tandfonline.com/doi/abs/10.1080/1351847X.2014.997369)
6. Cheema & Nartea, "Momentum returns, market states, and financial crisis: evidence from China and Hong Kong" — [core.ac.uk](https://core.ac.uk/download/185458184.pdf)
7. Ang, Hodrick, Xing & Zhang (2006), "The Cross-Section of Volatility and Expected Returns", Journal of Finance — [onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00836.x)
8. Ang, Hodrick, Xing & Zhang (2009), "High idiosyncratic volatility and low returns: International and further U.S. evidence", Journal of Financial Economics — [business.columbia.edu](https://business.columbia.edu/faculty/research/high-idiosyncratic-volatility-and-low-returns-international-and-further-us)
9. Frazzini & Pedersen (2014), "Betting Against Beta", Journal of Financial Economics — [nber.org](https://www.nber.org/papers/w18801)
10. Baker, Bradley & Wurgler (2011), "Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly", Financial Analysts Journal — [nber.org](https://www.nber.org/papers/w15467)
11. Hsu, Viswanathan, Wang & Wool (2018), "Anomalies in Chinese A-Shares", Journal of Portfolio Management — [proquest.com](https://www.proquest.com/openview/edd20f1a02228564319673d513e8e220/)
12. Acadian Asset Management, "Risk Mispricing in China A-Shares" — [acadian-asset.com](https://www.acadian-asset.com/-/media/files/thematic-research-paper-pdfs/acadian---risk-mispricing-in-china-a-shares.pdf)
13. Su (2025), "Calm Stocks, Wild Hopes: Explaining the Low-Volatility Anomaly" (NYU Shanghai thesis, CSMAR data) — [shanghai.nyu.edu](https://cdn.shanghai.nyu.edu/sites/default/files/honorsthesis2025_zijin_su.pdf)
14. Fama & French (1993), "Common risk factors in the returns on stocks and bonds", Journal of Financial Economics — [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/0304405X93900235)
15. Liu, Stambaugh & Yuan (2019), "Size and Value in China", Journal of Financial Economics — [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0304405X19300625)
16. Hu, Chen, Shao & Wang (2019), "Fama-French in China: Size and Value Factors in Chinese Stock Returns", International Review of Finance — [onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/10.1111/irfi.12177)
17. Novy-Marx (2013), "The Other Side of Value: The Gross Profitability Premium", Journal of Financial Economics — [nber.org](https://www.nber.org/papers/w15942)
18. Asness, Frazzini & Pedersen (2019), "Quality minus junk", Review of Accounting Studies — [springer.com](https://rd.springer.com/article/10.1007/s11142-018-9470-2)
19. Kakushadze (2016), "101 Formulaic Alphas", Wilmott (factor construction conventions) — [arxiv.org](https://arxiv.org/abs/1601.00991)
20. "Backtest Pitfalls in China A-Share Research: Six Failure Modes" (practitioner guide) — [stock.yuchenxu.cn](https://stock.yuchenxu.cn/en/guide/quant-backtest-pitfalls-china-a-share)
21. "Which factors in China? A pre-registered study", Journal of Empirical Finance (2025) — [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0927538X2500349X)
