# exp_001

Predictive Coding Network experiment on a combined dataset:
- EMNIST (byclass)
- FashionMNIST
- CIFAR100

This experiment aims to evaluate the sample efficiency of Predictive Coding Networks (PCN) compared to deep learning models like ResNet. We train each model using only 450 samples per class from each dataset, with similar network parameters. We then compare their performance on the test set and also analyze additional aspects such as training time and other relevant factors.

All images are converted to `3x32x32`, labels are offset into a single shared label space, and training/testing are done from `train.ipynb`.

## Files
- `train.ipynb` — main notebook (data prep, loaders, PCN model, train/test)
- `data/` — downloaded datasets (not tracked)
- `checkpoints/` — saved model weights (not tracked)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib seaborn tqdm