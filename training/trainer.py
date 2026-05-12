from __future__ import annotations

import torch
from tqdm import tqdm

from utils.metrics import compute_metrics, plot_confusion_matrix


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, mixed_precision: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        amp_enabled = mixed_precision and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def train_epoch(self, loader, max_steps: int | None = None):
        self.model.train()
        total_loss = 0.0
        step = 0
        for batch in tqdm(loader, desc="Train", leave=False):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                outputs = self.model(pixel_values=images, labels=labels)
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if self.scheduler:
            self.scheduler.step()
        denom = max(step, 1) if max_steps is not None else max(len(loader), 1)
        return total_loss / denom

    @torch.no_grad()
    def validate(self, loader, class_names, cm_out_path: str | None = None, max_steps: int | None = None):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        step = 0

        for batch in tqdm(loader, desc="Val", leave=False):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(pixel_values=images, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            step += 1
            if max_steps is not None and step >= max_steps:
                break

        metrics = compute_metrics(all_labels, all_preds)
        if cm_out_path:
            plot_confusion_matrix(metrics.confusion, class_names, cm_out_path)
        denom = max(step, 1) if max_steps is not None else max(len(loader), 1)
        return total_loss / denom, metrics
