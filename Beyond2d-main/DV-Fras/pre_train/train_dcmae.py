import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import PairedXRayDataset
from models_dcmae import dcmae_vit_small_patch16

# --------------------------------------------------------
# Configuration used by the manuscript-consistent model
# --------------------------------------------------------
IMG_SIZE = (224, 224)
IN_CHANS = 1
BATCH_SIZE = 16
LR = 1.0e-4
EPOCHS = 200
MASK_RATIO = 0.60
LAMBDA_CONSIST = 0.10
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
VIS_N = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_transform(image):
    image = image.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_vis(image, reconstruction, mask, save_path):
    """
    image, reconstruction, mask: [H, W]
    The binary mask follows the model convention: 0=visible, 1=masked.
    """
    image = image.detach().cpu().numpy()
    reconstruction = reconstruction.detach().cpu().clamp(0.0, 1.0).numpy()
    mask = mask.detach().cpu().numpy()

    masked_image = image * (1.0 - mask) + 0.5 * mask
    composed = image * (1.0 - mask) + reconstruction * mask

    panels = [image, masked_image, reconstruction, composed]
    panels = [Image.fromarray((panel.clip(0.0, 1.0) * 255).astype(np.uint8)) for panel in panels]
    canvas = Image.new("L", (sum(panel.width for panel in panels), panels[0].height))
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    canvas.save(save_path)


def mask_to_image(mask, model):
    """Expand a patch mask [L] to a pixel mask [H, W]."""
    grid_h, grid_w = model.patch_embed.grid_size
    patch_h, patch_w = model.patch_embed.patch_size
    if mask.numel() != grid_h * grid_w:
        raise ValueError(
            f"Mask contains {mask.numel()} elements, expected {grid_h * grid_w}."
        )
    mask = mask.reshape(grid_h, grid_w)
    return mask.repeat_interleave(patch_h, 0).repeat_interleave(patch_w, 1)


@torch.no_grad()
def visualize_results(model, val_dataset, epoch, save_dir):
    model.eval()
    vis_dir = save_dir / f"vis_epoch_{epoch}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    num_examples = min(VIS_N, len(val_dataset))
    if num_examples == 0:
        return
    indices = np.linspace(0, len(val_dataset) - 1, num_examples, dtype=int)

    for index in indices:
        item = val_dataset[index]
        if len(item) != 4:
            raise ValueError(
                "The validation dataset must return "
                "(ap, lat, ap_path, lat_path) when return_path=True."
            )
        ap, lat, ap_path, _ = item
        ap_batch = ap.unsqueeze(0).to(DEVICE, non_blocking=True)
        lat_batch = lat.unsqueeze(0).to(DEVICE, non_blocking=True)

        loss, pred_ap, pred_lat, mask_ap, mask_lat = model(
            ap_batch,
            lat_batch,
            mask_ratio=MASK_RATIO,
            lambda_consist=LAMBDA_CONSIST,
        )
        del loss

        recon_ap = model.unpatchify(pred_ap)
        recon_lat = model.unpatchify(pred_lat)
        mask_ap_image = mask_to_image(mask_ap[0], model)
        mask_lat_image = mask_to_image(mask_lat[0], model)

        name = Path(ap_path).stem
        save_vis(
            ap_batch[0, 0],
            recon_ap[0, 0],
            mask_ap_image,
            vis_dir / f"{name}_AP.png",
        )
        save_vis(
            lat_batch[0, 0],
            recon_lat[0, 0],
            mask_lat_image,
            vis_dir / f"{name}_LAT.png",
        )


def _unpack_training_batch(batch):
    if len(batch) < 2:
        raise ValueError("Each training batch must contain paired AP and LAT images.")
    return batch[0], batch[1]


def _run_epoch(model, loader, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    totals = {
        "loss": 0.0,
        "loss_rec_ap": 0.0,
        "loss_rec_lat": 0.0,
        "loss_consist": 0.0,
    }

    iterator = tqdm(loader, leave=False) if is_training else loader
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in iterator:
            ap, lat = _unpack_training_batch(batch)
            ap = ap.to(DEVICE, non_blocking=True)
            lat = lat.to(DEVICE, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            loss, _, _, _, _, details = model(
                ap,
                lat,
                mask_ratio=MASK_RATIO,
                lambda_consist=LAMBDA_CONSIST,
                return_details=True,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected: {loss.item()}")

            if is_training:
                loss.backward()
                optimizer.step()

            batch_values = {
                "loss": loss,
                "loss_rec_ap": details["loss_rec_ap"],
                "loss_rec_lat": details["loss_rec_lat"],
                "loss_consist": details["loss_consist"],
            }
            for name, value in batch_values.items():
                totals[name] += value.detach().item()

            if is_training:
                iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

    if len(loader) == 0:
        raise ValueError("The data loader is empty.")
    return {name: value / len(loader) for name, value in totals.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, path, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "img_size": IMG_SIZE,
                "patch_size": model.patch_embed.patch_size,
                "in_chans": IN_CHANS,
                "mask_ratio": MASK_RATIO,
                "lambda_consist": LAMBDA_CONSIST,
                "learning_rate": args.lr,
                "min_learning_rate": args.min_lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train the manuscript DC-MAE model.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    parser.add_argument("--lambda-consist", type=float, default=LAMBDA_CONSIST)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--visualize-every", type=int, default=10)
    return parser.parse_args()


def train(args):
    global MASK_RATIO, LAMBDA_CONSIST
    MASK_RATIO = args.mask_ratio
    LAMBDA_CONSIST = args.lambda_consist
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in (args.train_json, args.val_json):
        if not json_path.is_file():
            raise FileNotFoundError(f"Split file not found: {json_path}")

    train_dataset = PairedXRayDataset(str(args.train_json), transform=image_transform)
    val_dataset = PairedXRayDataset(
        str(args.val_json),
        transform=image_transform,
        return_paths=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = dcmae_vit_small_patch16(
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
    ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    history = []
    best_val_loss = float("inf")
    print(f"Start training DC-MAE for {args.epochs} epochs on {DEVICE}.")
    print(
        f"Mask ratio={MASK_RATIO:.2f}, "
        f"lambda_consist={LAMBDA_CONSIST:.2f}, image size={IMG_SIZE}."
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader)
        scheduler.step()

        record = {"epoch": epoch}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(record)
        with open(args.output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"val_rec_ap={val_metrics['loss_rec_ap']:.4f} | "
            f"val_rec_lat={val_metrics['loss_rec_lat']:.4f} | "
            f"val_cons={val_metrics['loss_consist']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), args.output_dir / "best_model_state_dict.pth")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                args.output_dir / "best_training_checkpoint.pth",
                args,
            )

        if args.visualize_every > 0 and (epoch % args.visualize_every == 0 or epoch == args.epochs):
            torch.save(
                model.state_dict(),
                args.output_dir / f"model_state_dict_ep{epoch}.pth",
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                args.output_dir / f"training_checkpoint_ep{epoch}.pth",
                args,
            )
            visualize_results(model, val_dataset, epoch, args.output_dir)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train(parse_args())
