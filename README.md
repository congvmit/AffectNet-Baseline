# AffectNet Baseline using Pytorch Lightning

Cong M. Vo


## Installation

```bash
pip install -r requirements.txt
```

## Training

1. Train baseline

```bash
python train.py -c experiments/baseline.yaml --gpus 1 --num_workers 6
```

2. Train with Vit

```bash
python train.py -c experiments/vit.yaml --gpus 1 --num_workers 6
```