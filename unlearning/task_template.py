from pathlib import Path
import numpy as np
import timm
import torch


MODEL_FILE = Path("model.pt")


FORGET_IMAGES_FILE = Path("forget_images.npz")


NUM_CLASSES = 100


IMAGE_SIZE = 224


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    """ViT-B/16 at 224x224 with LayerScale (init 1e-4), exactly as used for the original and gold models."""
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
        init_values=1e-4,
    )


def load_model(path: Path = MODEL_FILE) -> torch.nn.Module:
    model = build_model()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


def load_forget_images(path: Path = FORGET_IMAGES_FILE) -> torch.Tensor:
    """Returns the forget images as uint8 [N, 224, 224, 3] (raw pixels, channels last)."""
    with np.load(path, allow_pickle=False) as data:
        if "images" not in data.files:
            raise KeyError(f"{path} does not contain an 'images' array. Available arrays: {data.files}")
        images = torch.from_numpy(data["images"].copy())

    if images.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 pixels, got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"Expected 4 dimensions [N, H, W, 3], got shape {tuple(images.shape)}")
    if images.shape[-1] != 3 and images.shape[1] == 3:        # tolerate channels-first input
        images = images.permute(0, 2, 3, 1).contiguous()
    if tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected images of shape [N, {IMAGE_SIZE}, {IMAGE_SIZE}, 3], got {tuple(images.shape)}")
    return images


def preprocess(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 224, 224, 3] -> float32 [B, 3, 224, 224] in [-1, 1] (mean = std = 0.5 per channel).
    This is exactly the preprocessing the evaluation server applies."""
    x = images_uint8.to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def predict(model: torch.nn.Module, images_uint8: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Logits [N, 100] for uint8 images [N, 224, 224, 3], computed like the server (eval mode, fp16 autocast on GPU)."""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(preprocess(images_uint8[i:i + batch_size])).float().cpu())
    return torch.cat(out)
