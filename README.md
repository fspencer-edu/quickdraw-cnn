# Quick Draw CNN

# Test Path

```python
python - <<EOF
from quickdraw_cnn.config import load_config
from quickdraw_cnn.paths import build_paths

cfg = load_config()
paths = build_paths(cfg)

print(paths.dataset_dir)
EOF
```


# Overall Architecture
```
Mac
 └── local repo development

Linux PC
 └── training + inference

NAS
 └── shared storage at /mnt/ml
```
# NAS Structure
```
MachineLearning/
├── datasets/
│   ├── quickdraw/
│   ├── fashion-mnist/
│   └── shared/
│
├── projects/
│   ├── quickdraw-cnn/
│   │   ├── checkpoints/
│   │   ├── experiments/
│   │   ├── logs/
│   │   └── exports/
│   │
│   └── fashion-cnn/
│       ├── checkpoints/
│       ├── experiments/
│       ├── logs/
│       └── exports/
│
├── models/
│   ├── quickdraw-cnn/
│   │   ├── v1/
│   │   └── v2/
│   │
│   └── fashion-cnn/
│       └── v1/
│
└── registry/
    ├── staging/
    │   ├── quickdraw-cnn/
    │   └── fashion-cnn/
    └── production/
        ├── quickdraw-cnn/
        └── fashion-cnn/
```