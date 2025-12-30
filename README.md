# ModelTrace

**ModelTrace** is a research-grade framework for inspecting, debugging, and stress-testing machine learning models.  
It provides tools to analyze model sparsity, activations, distribution shifts, layer sensitivity, drift, and robustness metrics.

---

## Features

- ✅ Sparsity Analysis (Conv & FC layers)  
- ✅ Activation Statistics (mean, std, max)  
- ✅ Distribution Shift Stress Test  
- ✅ Activation Drift  
- ✅ Cross-Model Regression  
- ✅ Layer Sensitivity Analysis  
- ✅ Failure Attribution  
- ✅ Model Robustness Score  

---

## Installation

```bash
git clone https://github.com/MANISH-K-07/ModelTrace.git
cd ModelTrace
pip install -r requirements.txt
python setup.py install
```

---

## Usage
### Import API

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