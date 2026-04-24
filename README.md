# Mini Transformer Benchmark

Implementation of a Transformer encoder from scratch for binary sequence classification.

## Task
Given a sequence from vocabulary {PAD, A, B, C, D}, predict whether the first
non-padding token reappears in the second half of the non-padded portion.

## Files
- `data.py`       — Dataset loading and DataLoaders
- `model.py`      — Transformer components (attention, MHA, FFN, encoder, model)
- `train.py`      — Training and evaluation loops
- `benchmark.py`  — Runs all 4 variants, produces the benchmark table and plots
- `utils.py`      — Seeds, accuracy, plotting helpers
- `report.pdf`    — Full report

## How to run
1. Place `train.csv`, `validation.csv`, `test.csv` in the project root.
2. `pip install -r requirements.txt`
3. `python benchmark.py`

## Results
| Model | PE  | Heads | Layers | Val Acc | Test Acc | Train Time | Params |
|-------|-----|-------|--------|---------|----------|------------|--------|
| A     | Yes | 1     | 1      | 0.9189  | 0.9033   | 0.64 min   | 33,922 |
| B     | Yes | 4     | 1      | 0.9766  | 0.9629   | 0.76 min   | 33,922 |
| C     | No  | 4     | 1      | 0.8096  | 0.8213   | 0.67 min   | 33,922 |
| D     | Yes | 4     | 2      | 0.9844  | 0.9785   | 1.42 min   | 67,394 |

## Requirements
See `requirements.txt`.

## Author
[Your Name]# Mini Transformer Benchmark

Implementation of a Transformer encoder from scratch for binary sequence classification.

## Task
Given a sequence from vocabulary {PAD, A, B, C, D}, predict whether the first
non-padding token reappears in the second half of the non-padded portion.

## Files
- `data.py`       — Dataset loading and DataLoaders
- `model.py`      — Transformer components (attention, MHA, FFN, encoder, model)
- `train.py`      — Training and evaluation loops
- `benchmark.py`  — Runs all 4 variants, produces the benchmark table and plots
- `utils.py`      — Seeds, accuracy, plotting helpers
- `report.pdf`    — Full report

## How to run
1. Place `train.csv`, `validation.csv`, `test.csv` in the project root.
2. `pip install -r requirements.txt`
3. `python benchmark.py`

## Results
| Model | PE  | Heads | Layers | Val Acc | Test Acc | Train Time | Params |
|-------|-----|-------|--------|---------|----------|------------|--------|
| A     | Yes | 1     | 1      | 0.9189  | 0.9033   | 0.64 min   | 33,922 |
| B     | Yes | 4     | 1      | 0.9766  | 0.9629   | 0.76 min   | 33,922 |
| C     | No  | 4     | 1      | 0.8096  | 0.8213   | 0.67 min   | 33,922 |
| D     | Yes | 4     | 2      | 0.9844  | 0.9785   | 1.42 min   | 67,394 |

## Requirements
See `requirements.txt`.

## Author
Kyaw Zin Thant
