# CIFAR-100 participation suites

Both runners use the same registered CIFAR-100 methods and launch each method
with all 50 two-class tasks. The first suite uses full participation; the
second removes one randomly selected client from both model broadcast and
update aggregation at each synchronization. PILORA performs its own internal
client selection, so its blackout is sampled once per internal aggregation.

Run from this checkout on the GPU host, after installing the repository
requirements with a CUDA-enabled PyTorch/torchvision build and preparing
`dataset/cifar100-classes/0.npy` through `99.npy` plus
`dataset/class_order/class_order_cifar100_100clients.npy`:

```bash
python dataset/cifar100_npy.py
python scripts/run_cifar100_suite.py --launch --gpu 0
python scripts/run_cifar100_dropout_suite.py --launch --gpu 0
```

The dataset preparation command downloads CIFAR-100 and writes the expected
class arrays while retaining the original downloaded/extracted dataset files.

Without `--launch`, either command only prints the job commands. Jobs run one
at a time on the selected GPU by default. Use `--max-parallel N` to run N
methods simultaneously only when the GPU has enough free memory. Each mode
writes logs and method state beneath a separate timestamped directory under
`runs/`; the suite does not clear shared result or dataset directories.
Existing method hyperparameters are retained except that the runner sets
`--nt 50` so every run traverses the full CIFAR-100 task sequence.

The ReFedPlus, FedFCIL, GLFC, and LANDER JSON files are CIFAR-100 launch
configurations derived from the repository's available settings and code
defaults; they are not claims of paper-reproduction hyperparameter fidelity.
PILORA requires the DINO ViT weights to be cached or downloadable through
`timm`. If the expected dataset arrays or a CUDA device are absent, launch
preflight stops before any training job starts. The runner does not install
packages, download data, or launch training unless `--launch` is supplied.
