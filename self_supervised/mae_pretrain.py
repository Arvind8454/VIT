from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTMAEForPreTraining

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset_loader import DataConfig, get_self_supervised_loader


def parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining for ViT")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="experiments/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", default="facebook/vit-mae-base")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max steps per epoch for quick smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = DataConfig(dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)
    loader = get_self_supervised_loader(cfg)

    model = ViTMAEForPreTraining.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = args.mixed_precision and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        step = 0
        for images in loader:
            pixel_values = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                outputs = model(pixel_values=pixel_values)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        scheduler.step()
        denom = max(step, 1) if args.max_steps is not None else max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss / denom:.4f}")

    save_dir = os.path.join(args.output_dir, "mae_pretrained")
    model.save_pretrained(save_dir)
    print(f"Saved MAE weights to: {save_dir}")


if __name__ == "__main__":
    main()
