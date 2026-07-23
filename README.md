# DV-FraS

Official implementation of **Dual-View Fracture Scoring (DV-FraS)** for mRUST assessment using paired AP and lateral radiographs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## DC-MAE pre-training

Prepare paired AP and lateral image paths:

```json
[
  {
    "ap": "global_crops/mouse001_ap.png",
    "lat": "global_crops/mouse001_lat.png"
  }
]
```

Run pre-training:

```bash
python DV-Fras/pre_train/train_dcmae.py \
  --train-json splits/pretrain_train.json \
  --val-json splits/pretrain_val.json \
  --output-dir runs/dcmae
```

## DV-FraS scoring

Each examination contains paired images, two ROIs per view, and four scores:

```json
[
  {
    "subject_id": "mouse001",
    "examination_id": "mouse001_day14",
    "ap": "global_crops/mouse001_day14_ap.png",
    "lat": "global_crops/mouse001_day14_lat.png",
    "ap_rois": [
      "rois/mouse001_day14_ap_1.png",
      "rois/mouse001_day14_ap_2.png"
    ],
    "lat_rois": [
      "rois/mouse001_day14_lat_1.png",
      "rois/mouse001_day14_lat_2.png"
    ],
    "scores": [2, 3, 2, 2]
  }
]
```

Train:

```bash
python DV-Fras/score/train_dvfras.py \
  --train-json splits/score_train.json \
  --val-json splits/score_val.json \
  --pretrained runs/dcmae/best_model_state_dict.pth \
  --output-dir runs/dvfras
```

Predict and evaluate:

```bash
python DV-Fras/score/predict_dvfras.py \
  --data-json splits/score_test.json \
  --checkpoint runs/dvfras/best_model.pth \
  --output runs/dvfras/test_predictions.json

python DV-Fras/score/evaluate.py runs/dvfras/test_predictions.json
```

## Repository layout

```text
DV-Fras/
├── pre_train/          # DC-MAE pre-training
└── score/              # DV-FraS training and evaluation
examples/               # Example JSON files
```

