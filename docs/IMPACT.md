# Impact Analysis — From Measured Results to Operational Outcomes

Every statement below is labeled by evidence type: **[Official source]**, **[Measured here]**, or **[Scenario — assumptions stated]**.

## 1. The problem is large and growing — [Official source]
- **FTC.** U.S. consumers reported losing **$12.5B** to fraud in 2024, up 25% year over year. The number of reports stayed flat while the share of reporters who lost money rose from 27% to 38%, meaning fraud is becoming *more effective*, not more frequent. (FTC Consumer Sentinel press release, Mar 2025.)
- **FBI IC3.** **$16.6B** in reported cybercrime losses in 2024, up 33%; cyber-enabled fraud accounted for $13.7B. (FBI Internet Crime Report 2024.)
- **GAO.** The federal government loses an estimated **$233–$521B per year** to fraud, based on FY2018–2022 data. (GAO-24-105833, Apr 2024.)

## 2. The technical gap this project addresses
| Gap | Why it matters | FinGuard-XAI response | Evidence |
|---|---|---|---|
| Transaction-by-transaction models miss coordinated fraud | Rings spread activity so each event looks normal | Streaming temporal graph + GNN over each event's neighborhood | **[Measured here]** PR-AUC 0.933 vs 0.701 for a strong non-graph model (5 seeds) |
| False positives overload analysts and block legitimate customers | Every false alert costs review time and customer trust | Relationship context sharpens decisions | **[Measured here]** FPR 0.60% vs 2.50% (≈4× fewer false alarms) at higher recall |
| Supervised models miss new attack types | Fraud tactics keep evolving (FTC, FBI above) | Dual-channel alerting with a fixed novelty budget | **[Measured here]** Never-seen fraud recall 5.9% → 33.5%, at +0.7 pp FPR |
| Decisions must be reviewable | Analysts and auditors need reasons | GNNExplainer subgraph for each alert | **[Measured here]** ~100 ms per explanation, off the scoring path |
| Must run inside payment authorization | Scoring must not delay payments | Bounded neighborhoods, subgraph inference | **[Measured here]** p95 4.7 ms per transaction on one CPU thread |

## 3. Operational translation — [Scenario — assumptions stated]
Notebook 06 converts the *measured*, base-rate-independent recall and false-positive rates into daily alert volume and intercepted fraud dollars for a hypothetical institution. The defaults are 1M transactions/day, a 0.1% fraud rate, $350 average loss, and $4 per alert review. All inputs are editable. **These are illustrations, not observed savings.**

## 4. Breadth of applicability
The method requires only pseudonymous entity identifiers and timestamps. It is therefore relevant to card issuers, payment processors, fintechs, e-commerce platforms, and public-program payment integrity (the GAO estimate above). The open-source release lets any institution evaluate it on its own data.

## 5. What is not yet shown
Performance on an institution's real production data; adversarial robustness; multi-seed results on the public IEEE-CIS benchmark (the pipeline is included). These are the next milestones.
