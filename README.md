![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-red)
![Status](https://img.shields.io/badge/Status-Research--Grade-orange)
![ML](https://img.shields.io/badge/Focus-Model%20Inspection%20%26%20Debugging-purple)
![License](https://img.shields.io/badge/License-MIT-green)
[![GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-brightgreen)](https://manish-k-07.github.io/ModelTrace/)

# ModelTrace

**ModelTrace** is a research-grade framework for inspecting, debugging, and stress-testing machine learning models.
It provides tools to analyze model sparsity, activations, distribution shifts, layer sensitivity, drift, and robustness metrics — helping researchers and practitioners understand **how models behave under different inputs and perturbations**.

---

## Motivation & Academic Relevance

Debugging and analyzing ML models is a critical research problem, especially for deep learning systems where failures are often **non-intuitive and hard to trace**.
ModelTrace addresses this by providing **quantitative metrics and structured analysis** of model internals:

- Layer-wise sparsity and activation statistics
- Sensitivity of layers to input perturbations
- Drift and robustness metrics
- Cross-model regression and failure attribution

This framework aligns with research topics in **ML reliability, interpretability, and model debugging**, showcasing **research-level problem-solving skills**.

---

## Key Features

| Feature | Description |
|---------|-------------|
| Sparsity Analysis | Detect unused weights in convolutional and fully-connected layers |
| Activation Statistics | Compute mean, std, and max of layer activations |
| Distribution Shift Test | Simulate input perturbations and evaluate activation changes |
| Activation Drift | Measure sensitivity between different input sets |
| Cross-Model Regression | Compare activations between two models for regression drift |
| Layer Sensitivity | Identify layers most critical to output changes |
| Failure Attribution | Quantify each layer's contribution to output shifts |
| Model Robustness Score | Aggregate robustness metric combining drift, sensitivity, and distribution shift |

---

## ModelTrace Working Process

```
Input Model → Activation Hook Registration → Forward Pass →
Compute Metrics:
     - Sparsity
     - Activation Statistics
     - Distribution Shift
     - Activation Drift
     - Cross-Model Regression
     - Layer Sensitivity
     - Failure Attribution
     - Robustness Score
→ Results
```

- All experiments are implemented in the **ModelInspector API**.
- Activations are captured via **forward hooks**.
- Metrics are computed in a **modular and extensible manner**, allowing easy extension for new experiments.

---

## Installation

```bash
git clone https://github.com/MANISH-K-07/ModelTrace.git
cd ModelTrace
pip install -r requirements.txt
python setup.py install
```

## Usage
### Using the ModelInspector API

```python
from modeltrace.core.inspector import ModelInspector
from simple_cnn import SimpleCNN
import torch

model = SimpleCNN()
inspector = ModelInspector(model)

x = torch.randn(1, 1, 8, 8)
conv_s, fc_s, total_s = inspector.inspect(x)
```

### Run All Experiments

```bash
python examples/run_all_experiments.py
```

---

## Example Output

```bash
============================================
     Running ALL ModelTrace Experiments
============================================

[Experiment 1] Sparsity Analysis
Conv sparsity (avg): 0.0000
FC sparsity (avg):   0.0000
--------------------------------------------

[Experiment 2] Activation Statistics
conv1: mean=-0.0926, std=0.6074, max=1.5994
conv2: mean=0.0355, std=0.2371, max=0.8379
fc: mean=0.0229, std=0.1267, max=0.1856
--------------------------------------------

[Experiment 3] Distribution Shift Test
Normal mean activation magnitude:  0.2797
Shifted mean activation magnitude: 0.3642
--------------------------------------------

[Experiment 4] Activation Drift
Drift score: 0.1093
--------------------------------------------

[Experiment 5] Model Regression Test
Regression drift score: 0.3458
--------------------------------------------

[Experiment 6] Layer Sensitivity Analysis
conv1: sensitivity=0.1984
conv2: sensitivity=0.0725
fc: sensitivity=0.0292
--------------------------------------------

[Experiment 7] Failure Attribution
Output shift magnitude: 0.0257
Layer-wise contribution:
conv1: 0.6609
conv2: 0.2325
fc: 0.1066
--------------------------------------------

[Experiment 8] Model Robustness Score
FINAL ROBUSTNESS SCORE: 0.0646
--------------------------------------------

✅ All experiments completed successfully.
```

---

## Folder Structure

```
ModelTrace/
├── modeltrace/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── hooks.py
│       ├── inspector.py
│       ├── drift.py
│       └── metrics.py
├── examples/
│   ├── simple_cnn.py
│   ├── experiment{1...8}.py
│   └── run_all_experiments.py
|
├── README.md
├── requirements.txt
└── setup.py
```

---

## Limitations

- Supports **only PyTorch models** currently.
- Examples use small CNNs; large models may require adaptation.
- Drift and robustness metrics are **heuristic-based**; could be extended with more sophisticated statistical methods.
- No support yet for **real-world large-scale datasets or production pipelines**.

---

## Future Enhancements

- Extend support to **transformers and large CNNs**.
- Add **visualizations** (activation heatmaps, drift plots).
- Enable **automatic pruning or retraining suggestions** based on metrics.
- Support additional frameworks (e.g., **TensorFlow, ONNX**).
- Implement **CLI with argument parsing** for selective experiment runs.

---

## License

This project is open-source and available under the MIT License.
