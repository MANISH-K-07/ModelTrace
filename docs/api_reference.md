# ModelInspector API Reference

## Class: `ModelInspector`

### Constructor

```python
inspector = ModelInspector(model)
```

- `model`: PyTorch model to inspect.

### Methods

`inspect(input_tensor)`

- Computes **sparsity** of convolutional and FC layers.
- Returns (`conv_sparsity`, `fc_sparsity`, `total_sparsity`).

`inspect_activations(input_tensor)`

- Returns mean, std, and max of activations per layer.

`stress_test_distribution_shift(normal_input, shifted_input)`

- Returns (`normal_mag`, `shifted_mag`) representing mean activation magnitudes.

`compute_drift(input_a, input_b)`

- Computes activation drift between two inputs.
- Returns a single float drift score.

`compute_model_regression(input_tensor, other_inspector)`

- Compares activations between two models.
- Returns regression drift score.

`compute_failure_attribution(input_tensor, perturbed_tensor)`

- Computes layer-wise contribution to output shift.
- Returns dict `{layer_name: contribution_score}`.

`compute_robustness_score(input_tensor, shifted_tensor)`

- Aggregates drift, layer sensitivity, and distribution shift into a single score.
- Returns `robustness_score`.