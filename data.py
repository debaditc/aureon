"""A small CONTROLLED corpus, not BEIR. Purpose: exercise the harness and expose
the dense-vs-sparse tradeoff on two query types. It is deliberately built so
each retriever has queries where it wins and queries where it whiffs, with
distractors designed to fool the other retriever. Absolute nDCG here is a
sanity signal, not a publishable result. Swap in BEIR via the same interface
(docs: list[str], queries: list[(text, rel_set, type)]) for real validation.
"""

DOCS = [
    # --- finance/tech corpus ---
    "The XR-2000 trading terminal returned error code E-417 during the overnight batch settlement run.",   # 0
    "Our new terminal hardware ships with a faster matching engine and lower tick-to-trade latency.",        # 1 (dense-similar to 0, no code)
    "Settlement failures in the clearing pipeline are often caused by mismatched trade identifiers.",        # 2
    "Reducing cloud infrastructure spend requires rightsizing instances and committing to reserved capacity.",# 3
    "We cut our AWS bill by forty percent after moving batch jobs to spot instances and pruning idle nodes.",# 4 (dense-rel to 'reduce cloud cost', low lexical overlap)
    "A guide to lowering compute expenditure on public cloud through autoscaling and workload scheduling.",  # 5
    "The FIX protocol session dropped after a heartbeat timeout on gateway GW-09.",                          # 6
    "Market data feed handlers must reconnect automatically when a session is disconnected.",                # 7 (dense-rel to 6)
    "Portfolio drawdown exceeded the risk limit and triggered an automatic position unwind.",               # 8
    "When losses breach a threshold the system liquidates holdings to cap further downside.",                # 9 (dense-rel to 8, no shared terms)
    "Ticket INC-5521 reports intermittent latency spikes on the order router during peak hours.",            # 10
    "Slow order routing at high volume degrades execution quality and increases slippage.",                  # 11 (dense-rel to 10)
    "The ESG report claims carbon neutrality but discloses no scope 3 emissions methodology.",              # 12
    "Vague sustainability pledges without measurable targets are a hallmark of greenwashing.",               # 13 (dense-rel to 12)
    "Config flag ENABLE_TCA_V2 must be set to true to activate transaction cost analysis.",                 # 14
    "Transaction cost analysis measures implicit trading costs like market impact and timing risk.",         # 15 (dense-rel to 14)
    "Counterparty CP-330 breached its credit exposure limit on the interest rate swap book.",               # 16
    "Rising exposure to a single counterparty concentrates risk and can threaten solvency.",                 # 17 (dense-rel to 16)
    "The Kubernetes pod for service pricing-svc was OOMKilled after a memory leak in the cache layer.",      # 18
    "Containers that exhaust their memory allocation are terminated and restarted by the orchestrator.",     # 19 (dense-rel to 18)
    "Backtest results overfit to the 2021 regime and failed to generalize to live trading.",                # 20
    "A strategy that looks great in-sample but collapses out-of-sample is likely curve-fit.",                # 21 (dense-rel to 20)
    "Invoice number INV-88421 was flagged by the anomaly detector as a duplicate payment.",                 # 22
    "Detecting repeated or fraudulent disbursements protects against accounts-payable leakage.",             # 23 (dense-rel to 22)
    "The quarterly filing 10-K was submitted late due to an audit adjustment on deferred revenue.",          # 24
    "Delayed regulatory disclosures can signal underlying accounting problems to investors.",                # 25 (dense-rel to 24)
    "Rate limiter on endpoint /v2/orders rejected requests with HTTP 429 during the flash rally.",           # 26
    "Throttling protects a service from overload but can drop legitimate traffic under bursts.",             # 27 (dense-rel to 26)
    "The liquidity provider widened spreads sharply during the volatility spike.",                           # 28
    "Market makers quote wider when uncertainty rises, raising the cost of crossing the book.",              # 29 (dense-rel to 28)
    "A dividend reinvestment plan compounds returns by buying additional shares automatically.",             # 30
    "Automatically plowing payouts back into equity accelerates long-run wealth accumulation.",              # 31 (dense-rel to 30)
    "The margin call was issued after the collateral value fell below the maintenance requirement.",         # 32
    "If posted assets lose value the broker demands more funds to keep the position open.",                  # 33 (dense-rel to 32)
    "Latency arbitrage exploits stale quotes across venues within microseconds.",                            # 34
    "Speed-based strategies profit from price differences that exist only momentarily across exchanges.",    # 35 (dense-rel to 34)
]

# (query, relevant doc set, type)  type in {"lexical","semantic"}
QUERIES = [
    # LEXICAL: exact codes/entities. BM25 should win; dense should get distracted.
    ("XR-2000 error E-417", {0}, "lexical"),
    ("gateway GW-09 heartbeat timeout", {6}, "lexical"),
    ("ticket INC-5521 order router latency", {10}, "lexical"),
    ("ENABLE_TCA_V2 config flag", {14}, "lexical"),
    ("counterparty CP-330 credit limit", {16}, "lexical"),
    ("pricing-svc OOMKilled pod", {18}, "lexical"),
    ("invoice INV-88421 duplicate", {22}, "lexical"),
    ("endpoint /v2/orders HTTP 429", {26}, "lexical"),

    # SEMANTIC: paraphrase, low lexical overlap. Dense should win; BM25 should whiff.
    ("how to lower our public cloud costs", {3, 4, 5}, "semantic"),
    ("system sells off holdings when losses get too big", {8, 9}, "semantic"),
    ("company makes vague green claims it cannot back up", {12, 13}, "semantic"),
    ("broker demands more money when collateral drops", {32, 33}, "semantic"),
    ("strategy works on old data but fails live", {20, 21}, "semantic"),
    ("dealers quote wider when markets get choppy", {28, 29}, "semantic"),
    ("reinvesting payouts to grow wealth over time", {30, 31}, "semantic"),
    ("profiting from momentarily stale prices across venues", {34, 35}, "semantic"),
]
