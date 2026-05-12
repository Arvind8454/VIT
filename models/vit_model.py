import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTMAEModel


def build_vit_model(
    num_classes: int,
    model_name: str = "google/vit-base-patch16-224-in21k",
    from_mae_path: str | None = None,
    from_checkpoint: str | None = None,
):
    if from_checkpoint:
        return ViTForImageClassification.from_pretrained(from_checkpoint)

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if from_mae_path:
        mae_model = ViTMAEModel.from_pretrained(from_mae_path)
        model.vit.embeddings.load_state_dict(mae_model.embeddings.state_dict(), strict=False)
        model.vit.encoder.load_state_dict(mae_model.encoder.state_dict(), strict=False)
        model.vit.layernorm.load_state_dict(mae_model.layernorm.state_dict(), strict=False)
    return model


def build_processor(model_name_or_path: str = "google/vit-base-patch16-224-in21k"):
    return ViTImageProcessor.from_pretrained(model_name_or_path)


def move_to_device(model, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), device
