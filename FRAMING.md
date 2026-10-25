# Framing: mule detection at Rawaj Pay

## Decision

For each instant transfer, Rawaj Pay has to decide whether to let it through or hold it and send it to a fraud analyst. Dina's five analysts can only review a limited number of cases per day, so the real decision is which transfers deserve a human look first. The score only helps if it lets Dina draw a line that keeps the review queue small enough to be worked, while still catching most mule accounts.

## Metric

Accuracy is misleading because mules are a small share of transfers: a model that flags nothing looks excellent and catches nothing. I would use precision at k (k = what the analysts can review per day) so the queue is not full of false alarms, and recall at that same threshold so we know how many mules we still miss. To pick the line itself I would use expected cost, where a missed mule (cost_fn) is much more expensive than a wasted review (cost_fp), and cross-check it with `threshold_for_precision` when Dina wants a promise like "at least 60% of flagged cases are real".

## Baseline

The baseline to beat is a simple rule, not a model: for example, flag transfers that are large and sent from a new account, or rank by transfer amount only. A trivial "flag nothing" baseline is also useful as a reference because it has high accuracy and zero recall. Any model has to beat the rule at the same number of reviews per day, otherwise it is not worth the extra complexity.

## Classical ML or LLM

This is a labelled, structured, tabular problem, so classical ML (logistic regression first, then gradient-boosted trees) is the right tool: it is fast, cheap, easy to explain to analysts and to calibrate for a threshold. An LLM does not fit scoring individual transfers, but it could help later by writing a short plain-language summary of why a case was flagged. Before any model, the audit has to pass: leaks such as a column that only exists after a case is closed would make any score look perfect and useless in production.
