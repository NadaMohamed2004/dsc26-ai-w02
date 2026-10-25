# AI usage

## Tools used

I used Claude (Anthropic) to draft `rawaj/audit.py`, `rawaj/metrics.py` and the first version of `FRAMING.md`, and to write my extra tests in `tests/test_rawaj.py` (a brute-force comparison for `best_threshold` and `threshold_for_precision`, plus edge cases for the audit). I read and ran everything before submitting.

## Where the AI was wrong or incomplete

<!-- TODO (write this yourself, honestly): one real place where an AI answer was wrong, incomplete or different from the spec, and how you noticed (a test, the README, or reasoning). Example of the kind of thing to look for: the slides' snippet for best_threshold uses ascending np.unique + argmin, which keeps the LOWEST threshold on a tie, but the task says the HIGHEST. Only write it if it really happened to you. -->

## Function I wrote without AI


I wrote the body of audit() myself (the checks for missing values, duplicates, constant columns, labels, timestamps and leakage). I chose it because it is the main function of the task and I wanted to understand how each check works. The helper functions it calls (_missing_mask, _parse_times, _numeric_view, _auc) were written with AI help.
