from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ViTForImageClassification, ViTImageProcessor

from utils.device_utils import autocast_context, clear_device_cache, get_device, maybe_half_tensor, use_fp16
from xai.attention import attention_map_from_last_layer, attention_rollout, resize_attention_map, rollout_to_heatmap
from xai.gradcam import ViTGradCAM, overlay_heatmap
from xai.integrated_gradients import generate_integrated_gradients_map

PRETRAINED_MODEL_NAME = "google/vit-base-patch16-224"
CUSTOM_MODEL_PATH = os.path.join("experiments", "checkpoints", "best_model")
LLM_MODEL_NAME = "google/flan-t5-base"
DEVICE = get_device()
USE_FP16 = use_fp16(DEVICE)
ENABLE_INT8_QUANT = os.getenv("VIT_USE_INT8", "0") == "1"
LABEL_MAP = {
    "groenendael": "Belgian Shepherd dog",
    "tabby": "striped cat",
}

_MODEL_CACHE: dict[str, "PredictorBundle"] = {}
_CACHE_LOCK = threading.Lock()
_LLM_CACHE: "LLMBundle | None" = None
_LLM_LOCK = threading.Lock()
_LLM_LAST_FAIL_AT = 0.0
_LLM_FAIL_COOLDOWN_SEC = 20


@dataclass
class PredictorBundle:
    model: ViTForImageClassification
    processor: ViTImageProcessor
    class_names: list[str]
    gradcam: ViTGradCAM
    lock: threading.Lock
    source_name: str
    model_name: str
    quantized_model: ViTForImageClassification | None = None


@dataclass
class LLMBundle:
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    lock: threading.Lock


def _read_class_names(source: str, num_labels: int, id2label: dict | None) -> list[str]:
    if os.path.isdir(source):
        names_path = os.path.join(source, "class_names.txt")
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                return names
    id2label = id2label or {}
    return [id2label.get(i, str(i)) for i in range(num_labels)]


def _resolve_source(model_mode: str) -> tuple[str, str]:
    if model_mode == "custom" and os.path.isdir(CUSTOM_MODEL_PATH):
        return "custom", CUSTOM_MODEL_PATH
    return "pretrained", PRETRAINED_MODEL_NAME


def _load_model_and_processor(source: str) -> tuple[ViTForImageClassification, ViTImageProcessor]:
    """
    Load ViT model + processor with offline-first behavior.
    This avoids startup crashes when internet/DNS is unavailable.
    """
    errors: list[str] = []
    try:
        model = ViTForImageClassification.from_pretrained(source, local_files_only=True)
        processor = ViTImageProcessor.from_pretrained(source, local_files_only=True)
        return model, processor
    except Exception as exc:
        errors.append(f"offline cache load failed: {exc}")

    try:
        model = ViTForImageClassification.from_pretrained(source)
        processor = ViTImageProcessor.from_pretrained(source)
        return model, processor
    except Exception as exc:
        errors.append(f"online download failed: {exc}")
        raise RuntimeError(" | ".join(errors))


def _maybe_build_quantized_model(model: ViTForImageClassification) -> ViTForImageClassification | None:
    # Optional int8 path for CPU-only fast prediction.
    if str(DEVICE) != "cpu" or not ENABLE_INT8_QUANT:
        return None
    try:
        quant_source = copy.deepcopy(model).cpu().eval()
        quantized = torch.quantization.quantize_dynamic(quant_source, {torch.nn.Linear}, dtype=torch.qint8)
        return quantized
    except Exception:
        return None


def get_predictor_bundle(model_mode: str = "pretrained") -> PredictorBundle:
    cache_key, source = _resolve_source(model_mode)
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        effective_source = source
        effective_mode = cache_key
        try:
            model, processor = _load_model_and_processor(source)
        except Exception as exc:
            if source == PRETRAINED_MODEL_NAME and os.path.isdir(CUSTOM_MODEL_PATH):
                print(
                    f"[ModelLoad] Pretrained model unavailable ({exc}). Falling back to local custom checkpoint at {CUSTOM_MODEL_PATH}.",
                    flush=True,
                )
                effective_source = CUSTOM_MODEL_PATH
                effective_mode = "pretrained_fallback_custom"
                model, processor = _load_model_and_processor(effective_source)
            else:
                raise

        model.to(DEVICE)
        if USE_FP16:
            # Mixed-precision path for CUDA inference.
            model.half()
        model.eval()
        class_names = _read_class_names(
            effective_source,
            num_labels=model.config.num_labels,
            id2label=getattr(model.config, "id2label", None),
        )
        quantized_model = _maybe_build_quantized_model(model)
        bundle = PredictorBundle(
            model=model,
            processor=processor,
            class_names=class_names,
            gradcam=ViTGradCAM(model),
            lock=threading.Lock(),
            source_name=effective_mode,
            model_name=getattr(model, "name_or_path", effective_source),
            quantized_model=quantized_model,
        )
        _MODEL_CACHE[cache_key] = bundle
        return bundle


def preload_pretrained_model() -> None:
    """Force one-time model cache at startup to reduce first-request latency."""
    get_predictor_bundle("pretrained")


def _confidence_to_pct(confidence: float) -> float:
    return confidence * 100.0 if confidence <= 1.0 else confidence


def _friendly_label(label: str) -> str:
    return LABEL_MAP.get(label.lower().strip(), label)


def _shorten_explanation(text: str, max_sentences: int = 3) -> str:
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text.strip()) if c.strip()]
    if not chunks:
        return ""
    compact: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = chunk.lower()
        if key in seen:
            continue
        seen.add(key)
        compact.append(chunk)
        if len(compact) >= max_sentences:
            break
    return " ".join(compact)


def build_prompt(label: str, confidence: float) -> str:
    confidence_pct = _confidence_to_pct(confidence)
    friendly_label = _friendly_label(label)
    return (
        "You are an AI teacher helping a student understand an image classification result.\n\n"
        f"The model predicted the image is: {friendly_label}\n"
        f"Confidence: {confidence_pct:.2f}%\n\n"
        "Explain clearly:\n"
        "1. What this object/animal is in simple words\n"
        "2. What visual features (color, shape, texture, parts) the model likely used\n"
        "3. What the confidence level means\n\n"
        "Also mention that highlighted regions show where the model focused.\n\n"
        "Keep the explanation short (2-3 lines), simple, and non-technical."
    )


def _fallback_explanation(label: str, confidence: float) -> str:
    confidence_pct = _confidence_to_pct(confidence)
    friendly_label = _friendly_label(label)
    return (
        f"This image is predicted as {friendly_label} with {confidence_pct:.2f}% confidence. "
        "The model focused on key visible features in the highlighted regions."
    )


def get_llm_bundle() -> LLMBundle | None:
    global _LLM_CACHE, _LLM_LAST_FAIL_AT
    with _LLM_LOCK:
        if _LLM_CACHE is not None:
            return _LLM_CACHE
        if (time.time() - _LLM_LAST_FAIL_AT) < _LLM_FAIL_COOLDOWN_SEC:
            return None
        try:
            # Prefer local cache for fast startup and offline operation.
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME, local_files_only=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
                model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
            except Exception as exc:
                print(f"[LLM] Failed to load {LLM_MODEL_NAME}: {exc}", flush=True)
                _LLM_LAST_FAIL_AT = time.time()
                return None
        try:
            model.to(DEVICE)
            if USE_FP16:
                model.half()
            model.eval()
            _LLM_CACHE = LLMBundle(model=model, tokenizer=tokenizer, lock=threading.Lock())
            return _LLM_CACHE
        except Exception as exc:
            print(f"[LLM] Failed to initialize model on device: {exc}", flush=True)
            _LLM_LAST_FAIL_AT = time.time()
            return None


def generate_llm_explanation(label: str, confidence: float) -> str:
    prompt = build_prompt(label=label, confidence=confidence)
    print("Prompt:", prompt, flush=True)
    bundle = get_llm_bundle()
    if bundle is None:
        return _fallback_explanation(label, confidence)

    try:
        with bundle.lock:
            encoded = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
            if USE_FP16:
                input_ids = input_ids.long()
            with torch.inference_mode():
                with autocast_context(DEVICE, enabled=USE_FP16):
                    generated = bundle.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_beams=1,
                        repetition_penalty=1.1,
                    )
        print("Generated IDs:", generated, flush=True)
        text = bundle.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        print("Decoded:", text, flush=True)
        text = " ".join(text.split())
        text = _shorten_explanation(text, max_sentences=3)
        if text.strip() == "":
            return _fallback_explanation(label, confidence)
        confidence_pct = _confidence_to_pct(confidence)
        if "highlight" not in text.lower():
            text = f"{text} The highlighted regions show where the model focused."
        if "%" not in text:
            text = f"{text} Confidence ({confidence_pct:.2f}%) indicates how sure the model is."
        return text
    except Exception:
        return _fallback_explanation(label, confidence)


def _image_to_pixel_values(image: Image.Image, processor: ViTImageProcessor, resize_224: bool = False, for_grad: bool = False) -> torch.Tensor:
    if resize_224:
        image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    if USE_FP16 and not for_grad:
        pixel_values = maybe_half_tensor(pixel_values, DEVICE, enabled=True)
    else:
        pixel_values = pixel_values.float()
    return pixel_values


def _run_model_forward(
    bundle: PredictorBundle,
    pixel_values: torch.Tensor,
    return_attentions: bool = False,
    use_quantized_path: bool = False,
):
    model = bundle.quantized_model if use_quantized_path and bundle.quantized_model is not None else bundle.model
    prev_impl = getattr(bundle.model.config, "_attn_implementation", None)
    switched_impl = bool(return_attentions and prev_impl is not None and prev_impl != "eager")
    if switched_impl:
        bundle.model.config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            with autocast_context(DEVICE, enabled=USE_FP16):
                outputs = model(
                    pixel_values=pixel_values,
                    output_attentions=return_attentions,
                    return_dict=True,
                )
    finally:
        if switched_impl:
            bundle.model.config._attn_implementation = prev_impl
    return outputs


def _topk_from_logits(logits: torch.Tensor, class_names: list[str], k: int = 3) -> tuple[int, float, list[dict]]:
    probs = torch.softmax(logits.float(), dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_idx].item())
    top_k = min(k, probs.shape[1])
    top_vals, top_idxs = torch.topk(probs, k=top_k, dim=1)
    top_predictions: list[dict] = []
    for idx, val in zip(top_idxs[0].tolist(), top_vals[0].tolist()):
        label = class_names[idx] if idx < len(class_names) else str(idx)
        top_predictions.append({"label": label, "confidence": float(val), "pred_idx": int(idx)})
    return pred_idx, confidence, top_predictions


def _get_spatial_focus(heatmap: np.ndarray | None) -> str:
    if heatmap is None:
        return "undetermined region"
    h, w = heatmap.shape
    ys, xs = np.where(heatmap >= np.percentile(heatmap, 85))
    if len(xs) == 0 or len(ys) == 0:
        return "diffuse spatial focus"
    x_mean = xs.mean() / max(w, 1)
    y_mean = ys.mean() / max(h, 1)
    horizontal = "left" if x_mean < 0.33 else ("right" if x_mean > 0.66 else "center")
    vertical = "upper" if y_mean < 0.33 else ("lower" if y_mean > 0.66 else "middle")
    return f"{vertical}-{horizontal} region"


def _confidence_interpretation(confidence: float) -> str:
    pct = _confidence_to_pct(confidence)
    if pct >= 85:
        return "High confidence indicates strong alignment with discriminative features."
    if pct >= 60:
        return "Moderate confidence suggests partial feature overlap with neighboring classes."
    return "Lower confidence indicates ambiguity and potentially overlapping visual cues."


def _build_technical_explanation(
    result: dict,
    gradcam_map: np.ndarray | None,
    attn_map: np.ndarray | None,
    rollout_map: np.ndarray | None,
) -> str:
    label = result["label"]
    confidence = result["confidence"]
    conf_pct = _confidence_to_pct(confidence)
    focus_region = _get_spatial_focus(gradcam_map if gradcam_map is not None else attn_map)
    return (
        f"The model predicts '{label}' with {conf_pct:.2f}% confidence. "
        f"Grad-CAM highlights salient regions with strong feature attribution, mainly around the {focus_region}. "
        "Attention maps reveal patch-level attention distribution, while attention rollout aggregates multi-layer context for global reasoning. "
        f"{_confidence_interpretation(confidence)}"
    )


def _build_contrastive_explanation(result: dict) -> str:
    top_predictions = result.get("top_predictions", [])
    if len(top_predictions) < 2:
        return "Contrastive explanation unavailable because a second candidate class was not found."
    top1 = top_predictions[0]
    top2 = top_predictions[1]
    delta = _confidence_to_pct(top1["confidence"] - top2["confidence"])
    return (
        f"The model favors '{top1['label']}' over '{top2['label']}' by {delta:.2f} confidence points. "
        f"This suggests stronger discriminative features and spatial focus for '{top1['label']}', "
        f"while '{top2['label']}' shares partial but weaker feature alignment."
    )


def predict_image(
    image: Image.Image,
    model_mode: str = "pretrained",
    resize_224: bool = False,
    return_attentions: bool = False,
) -> dict:
    """
    Predict image class using shared ViT pipeline.
    Keeps backward-compatible output keys while adding top-k and model metadata.
    """
    bundle = get_predictor_bundle(model_mode)
    pixel_values = _image_to_pixel_values(image, bundle.processor, resize_224=resize_224, for_grad=False)

    with bundle.lock:
        # Quantized model is used only for quick non-attention prediction.
        use_quantized = bool(not return_attentions)
        outputs = _run_model_forward(
            bundle=bundle,
            pixel_values=pixel_values,
            return_attentions=return_attentions,
            use_quantized_path=use_quantized,
        )

    pred_idx, confidence, top_predictions = _topk_from_logits(outputs.logits, bundle.class_names, k=3)
    label = bundle.class_names[pred_idx] if pred_idx < len(bundle.class_names) else str(pred_idx)

    return {
        "label": label,
        "confidence": confidence,
        "pred_idx": pred_idx,
        "pixel_values": pixel_values,
        "outputs": outputs,
        "class_names": bundle.class_names,
        "top_predictions": top_predictions,
        "model_mode": bundle.source_name,
        "model_name": bundle.model_name,
        "num_classes": len(bundle.class_names),
        "device": str(DEVICE),
        "fp16_enabled": USE_FP16,
        "bundle": bundle,
    }


def explain_image(
    image: Image.Image,
    model_mode: str = "pretrained",
    include_gradcam: bool = True,
    include_attention_map: bool = True,
    include_rollout: bool = True,
    include_llm_explanation: bool = True,
    include_integrated_gradients: bool = False,
    resize_224: bool = False,
) -> dict:
    """
    Run explainability flow on top of prediction output.
    Supports Grad-CAM, attention map, rollout, optional IG, structured and contrastive text.
    """
    pred = predict_image(image, model_mode=model_mode, resize_224=resize_224, return_attentions=True)
    image_np = np.array(image)
    outputs = pred["outputs"]
    bundle = pred["bundle"]
    pred_idx = pred["pred_idx"]

    # Gradient-based methods are more stable in fp32.
    pixel_values_grad = _image_to_pixel_values(image, bundle.processor, resize_224=resize_224, for_grad=True)

    result = {
        "label": pred["label"],
        "confidence": pred["confidence"],
        "pred_idx": pred_idx,
        "model_mode": pred["model_mode"],
        "model_name": pred["model_name"],
        "num_classes": pred["num_classes"],
        "device": pred["device"],
        "fp16_enabled": pred["fp16_enabled"],
        "top_predictions": pred["top_predictions"],
        "gradcam_overlay": None,
        "attention_map_overlay": None,
        "attention_rollout_overlay": None,
        "integrated_gradients_overlay": None,
        "patch_attention_scores": None,
        "explanation": None,
        "technical_explanation": None,
        "contrastive_explanation": None,
    }

    gradcam_map = None
    attention_map = None
    rollout_map = None

    if include_gradcam:
        with bundle.lock:
            cam = bundle.gradcam.generate(pixel_values_grad, class_idx=pred_idx)
        gradcam_map = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)) / 255.0
        result["gradcam_overlay"] = overlay_heatmap(image_np, gradcam_map)

    if include_attention_map:
        attention_map = attention_map_from_last_layer(outputs.attentions)
        if attention_map is not None:
            # Preserve patch-level map for future clickable patch UI.
            result["patch_attention_scores"] = attention_map.tolist()
            attn_resized = resize_attention_map(attention_map, image.size)
            result["attention_map_overlay"] = overlay_heatmap(image_np, attn_resized)

    if include_rollout:
        rollout_mask = attention_rollout(outputs.attentions)
        if rollout_mask is not None:
            grid_size = int((rollout_mask.size(-1)) ** 0.5)
            rollout_map = rollout_to_heatmap(rollout_mask[0], grid_size)
            rollout_resized = resize_attention_map(rollout_map, image.size)
            result["attention_rollout_overlay"] = overlay_heatmap(image_np, rollout_resized)

    if include_integrated_gradients:
        with bundle.lock:
            ig_map = generate_integrated_gradients_map(bundle.model, pixel_values_grad, pred_idx, steps=24)
        ig_resized = np.array(Image.fromarray((ig_map * 255).astype(np.uint8)).resize(image.size)) / 255.0
        result["integrated_gradients_overlay"] = overlay_heatmap(image_np, ig_resized)

    result["technical_explanation"] = _build_technical_explanation(result, gradcam_map, attention_map, rollout_map)
    result["contrastive_explanation"] = _build_contrastive_explanation(result)

    if include_llm_explanation:
        result["explanation"] = generate_llm_explanation(label=result["label"], confidence=result["confidence"])

    # Lightweight cleanup to avoid GPU memory growth in repeated sessions.
    clear_device_cache(DEVICE)
    return result


def generate_gradcam_for_class(
    image: Image.Image,
    model_mode: str,
    class_idx: int,
    resize_224: bool = False,
) -> np.ndarray:
    """
    Generate a Grad-CAM overlay for a specific target class.
    Useful for contrastive "why this vs why not" visual analysis.
    """
    bundle = get_predictor_bundle(model_mode)
    pixel_values_grad = _image_to_pixel_values(image, bundle.processor, resize_224=resize_224, for_grad=True)
    with bundle.lock:
        cam = bundle.gradcam.generate(pixel_values_grad, class_idx=class_idx)
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)) / 255.0
    return overlay_heatmap(np.array(image), cam_resized)
