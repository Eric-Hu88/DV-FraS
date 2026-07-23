import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from PIL import Image

from data_loader import FHADataset
from model_dvfras import FHAssessmentNet


def image_transform(image):
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DV-FraS for cortex-level mRUST scoring.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-lr", type=float, default=1.0e-5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lambda-reg", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--head-only-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_encoder_stage(model, epoch, args):
    encoder = model.encoder_global
    head_only = bool(model._l2sp_reference) and epoch < args.head_only_epochs
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = not head_only and name != "pos_embed"
    if head_only:
        return 0
    return len(encoder.blocks)


def parameter_groups(model, max_lr, layer_decay, weight_decay):
    depth = len(model.encoder_global.blocks)
    groups = []
    assigned = set()

    def add(parameters, lr, force_no_decay=False):
        parameters = [parameter for parameter in parameters if id(parameter) not in assigned]
        assigned.update(id(parameter) for parameter in parameters)
        decay = [] if force_no_decay else [parameter for parameter in parameters if parameter.ndim > 1]
        no_decay = parameters if force_no_decay else [parameter for parameter in parameters if parameter.ndim <= 1]
        for selected, group_decay in ((decay, weight_decay), (no_decay, 0.0)):
            if selected:
                groups.append(
                    {
                        "params": selected,
                        "lr": lr,
                        "initial_lr": lr,
                        "weight_decay": group_decay,
                    }
                )

    add(model.encoder_global.patch_embed.parameters(), max_lr * layer_decay ** (depth + 1))
    add([model.encoder_global.cls_token], max_lr * layer_decay ** (depth + 1), True)
    for index, block in enumerate(model.encoder_global.blocks):
        add(block.parameters(), max_lr * layer_decay ** (depth - index))
    add(model.encoder_global.norm.parameters(), max_lr)
    add(model.encoder_local.parameters(), max_lr)
    add(model.film.parameters(), max_lr)
    add(model.classifier.parameters(), max_lr)
    return groups


def classification_loss(logits, labels):
    per_cortex = F.cross_entropy(
        logits.reshape(-1, 4), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    return per_cortex.mean(dim=0).sum()


def run_epoch(model, loader, device, lambda_reg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_abs = total_correct = total_count = 0.0
    rows = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for globals_, locals_, labels, exam_ids, subject_ids in loader:
            globals_ = globals_.to(device, non_blocking=True)
            locals_ = locals_.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(globals_, locals_)
            ce = classification_loss(logits, labels)
            loss = ce + lambda_reg * model.get_l2_sp_loss()
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=-1)
            count = labels.numel()
            total_loss += loss.detach().item() * labels.shape[0]
            total_abs += (predictions - labels).abs().sum().item()
            total_correct += predictions.eq(labels).sum().item()
            total_count += count
            if not training:
                for batch_index, exam_id in enumerate(exam_ids):
                    rows.append(
                        {
                            "examination_id": exam_id,
                            "subject_id": subject_ids[batch_index],
                            "target": (labels[batch_index] + 1).cpu().tolist(),
                            "prediction": (predictions[batch_index] + 1).cpu().tolist(),
                        }
                    )
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": total_abs / total_count,
        "top1": total_correct / total_count,
    }, rows


def save_predictions(rows, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main(args):
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FHADataset(args.train_json, image_transform, image_transform)
    val_data = FHADataset(args.val_json, image_transform, image_transform)
    if not train_data or not val_data:
        raise ValueError("Training and validation datasets must both be non-empty.")
    overlap = train_data.subject_ids & val_data.subject_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Subject leakage between training and validation splits: {examples}")
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    model = FHAssessmentNet(args.pretrained).to(device)
    optimizer = AdamW(
        parameter_groups(model, args.max_lr, args.layer_decay, args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    best_mae = float("inf")
    epochs_without_improvement = 0
    metrics_path = args.output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("epoch", "unfrozen_blocks", "train_loss", "train_mae", "train_top1", "val_loss", "val_mae", "val_top1"),
        )
        writer.writeheader()
        for epoch in range(args.epochs):
            unfrozen = set_encoder_stage(model, epoch, args)
            train_metrics, _ = run_epoch(
                model, train_loader, device, args.lambda_reg, optimizer
            )
            val_metrics, predictions = run_epoch(
                model, val_loader, device, args.lambda_reg
            )
            scheduler.step()
            record = {
                "epoch": epoch + 1,
                "unfrozen_blocks": unfrozen,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            writer.writerow(record)
            metrics_file.flush()
            print(
                f"Epoch {epoch + 1:03d} | train MAE {train_metrics['mae']:.4f} | "
                f"val MAE {val_metrics['mae']:.4f} | val Top-1 {val_metrics['top1']:.4%}"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_without_improvement = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
                    args.output_dir / "best_model.pth",
                )
                save_predictions(predictions, args.output_dir / "best_val_predictions.json")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping after {epoch + 1} epochs. Best validation MAE: {best_mae:.6f}")
                    break


if __name__ == "__main__":
    main(parse_args())
