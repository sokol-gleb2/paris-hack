import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


TRANSFORMS = {
    "crop": {"box": [6, 6, 218, 218], "resize": [224, 224], "interpolation": "Pillow BILINEAR"},
    "brightness": {"factor": 1.05, "implementation": "Pillow ImageEnhance.Brightness"},
    "blur": {"radius": 0.5, "implementation": "Pillow GaussianBlur"},
    "flip": {"implementation": "Pillow FLIP_LEFT_RIGHT"},
}


def transform_batch(raw, name):
    output = []
    for pixels in raw:
        image = Image.fromarray(pixels)
        if name == "crop":
            image = image.crop(tuple(TRANSFORMS[name]["box"])).resize((224, 224), Image.Resampling.BILINEAR)
        elif name == "brightness":
            image = ImageEnhance.Brightness(image).enhance(1.05)
        elif name == "blur":
            image = image.filter(ImageFilter.GaussianBlur(0.5))
        elif name == "flip":
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        output.append(np.asarray(image))
    return torch.from_numpy(np.stack(output))
