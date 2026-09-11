# CoDFE

Code, pretrained models, and test samples for **CoDFE: Neighbor discriminator driven semi-supervised deep learning for accurate urban forest biomass estimation using satellite remote sensing imagery**.

The release contains eight ResNet18 CoDFE checkpoints, normalization statistics, and the 28 test patches used for the results in Table 6. Each patch has 39 channels and a 3 × 3 spatial window. The output feature dimension is 2.

## Install

Tested with Python 3.11. The eight checkpoints occupy about 692 MiB.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.8, use `https://download.pytorch.org/whl/cu128` instead of the CPU index when installing PyTorch. The scikit-learn version is pinned because the checkpoints contain fitted KNN regressors.

## Test

Run from the repository root:

```bash
python test.py
```

The default is the full-training-set checkpoint with pseudo labels, evaluated on CPU. The script prints only R2 and RMSE; RMSE is in t/ha. To choose a checkpoint or use a GPU:

```bash
python test.py --checkpoint weights/codfe_ori_4_4.pth
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint weights/codfe_pseudo_4_4.pth --device cuda:0
```

Testing keeps CSV order, calls `predict` with images only, and leaves the fitted KNN regressors unchanged. Predictions from the two branches are averaged. Both predictions and labels are converted from kg per 400 m² plot to t/ha using a factor of 0.025.

## Results

`ori` means without pseudo labels; `pseudo` means with pseudo labels. The fraction in a filename is the labeled training fraction, not the test fraction. Every checkpoint uses the same 28 test samples.

| Checkpoint | Paper R² | Paper RMSE | CPU R² | CPU RMSE |
|---|---:|---:|---:|---:|
| codfe_ori_1_4.pth | 0.2226 | 38.77 | 0.22255 | 38.77 |
| codfe_ori_2_4.pth | 0.2764 | 37.41 | 0.27637 | 37.41 |
| codfe_ori_3_4.pth | 0.2726 | 37.51 | 0.27263 | 37.51 |
| codfe_ori_4_4.pth | 0.4031 | 33.98 | 0.40309 | 33.98 |
| codfe_pseudo_1_4.pth | 0.2456 | 38.20 | 0.24507 | 38.21 |
| codfe_pseudo_2_4.pth | 0.2843 | 37.20 | 0.28425 | 37.20 |
| codfe_pseudo_3_4.pth | 0.1844 | 39.71 | 0.18444 | 39.71 |
| codfe_pseudo_4_4.pth | 0.5148 | 30.63 | 0.51799 | 30.53 |

All RMSE values are in t/ha. On an RTX 5090 with PyTorch 2.10.0 and CUDA 12.8, all eight checkpoints match the paper at its displayed precision. CPU testing uses the same checkpoint files and test data. The largest observed CPU difference is about 0.0032 in R² and 0.10 t/ha in RMSE. Small floating-point differences in learned features can change KNN neighbor selection; results may therefore vary slightly across devices and library builds.

## Train

```bash
python train.py
python test.py --checkpoint outputs/codfe.pth
```

The default training input is `data/release/test.csv`, so this is a runnable demonstration, not an independent test of a newly trained model. Use separate training and test data for a new experiment. The pretrained checkpoints above reproduce the paper's evaluation; retraining from the released samples does not reproduce the original training runs.

To use other labeled samples and enable pseudo label selection:

```bash
python train.py --data-dir data/release --train-csv train.csv \
  --unlabeled-csv unlabeled.csv --pseudo-iterations 10 --epochs 200 \
  --device cuda:0
```

CSV files contain `RegionIndex` and `TGB`; TIFF filenames match `RegionIndex`. `TGB` is plot biomass in kg. The unlabeled CSV may use zero placeholders for `TGB`; those values are not used in pseudo label selection. Keep the same 39-channel order and supply the matching normalization statistics. Training fits the two supervised branches, screens candidate pseudo labels by neighborhood error reduction, passes accepted labels to the other branch, and retrains on the expanded samples. New checkpoints are written to `outputs/`, leaving the released weights intact.

## Data and code availability

This is a partial release of the model code, pretrained checkpoints, normalization statistics, and test dataset. The complete dataset is available from the corresponding author, **Zhibo Chen** (zhibo@bjfu.edu.cn), upon reasonable request.

The original checkpoints include the feature vectors and reference labels needed by their fitted KNN regressors. They are loaded as PyTorch serialized objects; only load checkpoint files from a trusted source.
