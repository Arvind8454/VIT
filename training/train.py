from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vit_model import build_vit_model, build_processor, move_to_device
from utils.dataset_loader import DataConfig, get_dataloaders
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for ViT")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="experiments/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--use_mae_weights", default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max steps per epoch for quick smoke tests")
    parser.add_argument("--max_val_steps", type=int, default=None, help="Optional max validation/test steps")
    parser.add_argument("--skip_test", action="store_true", help="Skip final test evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join("experiments", "logs"), exist_ok=True)

    cfg = DataConfig(dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)

    model = build_vit_model(num_classes=len(class_names), model_name=args.model_name, from_mae_path=args.use_mae_weights)
    processor = build_processor()
    model, device = move_to_device(model)
    print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(model, optimizer, scheduler, device, mixed_precision=args.mixed_precision)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, max_steps=args.max_steps)
        val_loss, metrics = trainer.validate(
            val_loader,
            class_names,
            cm_out_path=os.path.join("experiments", "logs", "confusion_matrix.png"),
            max_steps=args.max_val_steps,
        )
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics.accuracy:.4f}")

        if metrics.accuracy > best_acc:
            best_acc = metrics.accuracy
            save_dir = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "class_names.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(class_names))
            processor.save_pretrained(save_dir)

    print(f"Best Val Acc: {best_acc:.4f}")

    if not args.skip_test:
        test_loss, test_metrics = trainer.validate(test_loader, class_names, max_steps=args.max_val_steps)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()
