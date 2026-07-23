import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loader import FHADataset
from model_dvfras import FHAssessmentNet
from train_dvfras import image_transform, run_epoch, save_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DV-FraS checkpoint.")
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FHADataset(args.data_json, image_transform, image_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model = FHAssessmentNet().to(device)
    model.load_state_dict(state)
    metrics, rows = run_epoch(model, loader, device, lambda_reg=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(rows, args.output)
    print(
        f"MAE {metrics['mae']:.6f} | Top-1 {metrics['top1']:.4%} | "
        f"predictions {args.output}"
    )


if __name__ == "__main__":
    main()
