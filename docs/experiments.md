# Experiments in ModelTrace

## Experiment 1: Sparsity Analysis
- Detects unused weights in convolutional and fully-connected layers.
- Provides insights for model pruning and optimization.

## Experiment 2: Activation Statistics
- Computes mean, standard deviation, and max of activations per layer.
- Identifies abnormal layer behavior.

## Experiment 3: Distribution Shift Stress Test
- Simulates input perturbations.
- Measures change in activations to evaluate robustness to input shifts.

## Experiment 4: Activation Drift
- Quantifies differences in activations between input sets.
- Highlights sensitivity to perturbations.

## Experiment 5: Cross-Model Regression
- Compares two models on the same input.
- Measures regression drift to detect model discrepancies.

## Experiment 6: Layer Sensitivity Analysis
- Measures which layers contribute most to output changes.
- Useful for understanding bottlenecks or critical layers.

## Experiment 7: Failure Attribution
- Quantifies how individual layers contribute to output shift.
- Identifies which layers are responsible for failures.

## Experiment 8: Model Robustness Score
- Aggregates multiple metrics into a single robustness score.
- Weighted combination of drift, sensitivity, and distribution shift.
