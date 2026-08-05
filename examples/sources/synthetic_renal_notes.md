# Synthetic renal reasoning notes

This file is deliberately synthetic and is provided only to exercise the corpus pipeline. It is not medical guidance.

## Repeated-state calculation pattern

When a deterministic clinical formula is evaluated at multiple time points, all input variables, units, and rounding conventions must be stated for each time point. The intervention between time points should be represented as an event, not assumed to be causal unless the source establishes causality.

## Decision-table construction pattern

A task based on a decision table should bind exactly one row after all conditions are considered. Boundary values must be represented explicitly. If two rows can match the same patient state, the generated question is ambiguous and must be rejected or refined.

## Evidence separation

Clinical thresholds and treatment recommendations should remain in a versioned evidence record. A construction skill may require a threshold, but should not hard-code a value that can change between guideline versions.

