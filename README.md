# Quick Draw CNN

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