# Synthetic Antigen V0 dataset card

This episode is entirely synthetic and exists only to test VaxReplay's contracts, scoring, and temporal-leakage controls. It is not derived from a real pathogen, publication, sequence, clinical record, or laboratory result, and it must not be used as scientific evidence.

The three candidate identifiers are opaque fictional antigen targets. The evidence contains high-level fictional observations but no sequences, constructs, experimental procedures, or manufacturing instructions.

`ev-future-canary` is intentionally collected before the decision cutoff but released afterward. A correct public view excludes it based on `available_at`, not `collected_at`.

License identifier: `synthetic-internal-test-data`. There are no upstream data sources.

