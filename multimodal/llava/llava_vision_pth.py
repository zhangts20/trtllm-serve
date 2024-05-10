import os
import json
import torch
import torch.nn.functional as F

from typing import Dict
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from utils import calculate_time


class LlavaVision(torch.nn.Module):

    def __init__(self, model_dir: str, dtype=torch.float16):
        super().__init__()

        config_file = os.path.join(model_dir, "config.json")
        config: Dict = json.load(open(config_file, "r"))

        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # the path of vision model and projector model
        self.vision_path = config.get('mm_vision_tower',
                                      'openai/clip-vit-large-patch14-336')
        if isinstance(self.vision_path, list):
            self.vision_path = self.vision_path[0]
        if self.vision_path.startswith("./"):
            self.vision_path = os.path.join(model_dir, self.vision_path)
        self.mm_path = os.path.join(model_dir, "mm_projector.bin")

        self.device = "cuda"
        self.dtype = dtype

        # load model
        self.load_clip()
        self.load_mm_projector()

    def load_clip(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_path).to(self.dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower = self.vision_tower.to(self.device)

    def load_mm_projector(self):
        self.mm_weights: Dict[str, torch.Tensor] = torch.load(self.mm_path)
        for k, v in self.mm_weights.items():
            self.mm_weights[k] = v.to(self.device)

    def load_image(self, image_dir: str, num=1):
        if os.path.isdir(image_dir):
            images = []
            for image in os.listdir(image_dir):
                images.append(Image.open(os.path.join(image_dir, image)))
        else:
            images = [Image.open(image_dir)]
        images = images[:num]

        processor = CLIPImageProcessor.from_pretrained(self.vision_path)
        image: torch.Tensor = processor.preprocess(
            images, return_tensors="pt")["pixel_values"]
        return image.to(dtype=self.dtype, device=self.device)

    @calculate_time
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vision_tower(
            x, output_hidden_states=True).hidden_states[self.select_layer]
        if self.select_feature == "patch":
            x = x[:, 1:].contiguous()

        B, L, N = x.shape
        x = x.view(-1, N)

        # mm projector, Linear + GELU + Linear
        x = F.linear(input=x,
                     weight=self.mm_weights["model.mm_projector.0.weight"].to(
                         self.dtype),
                     bias=self.mm_weights["model.mm_projector.0.bias"].to(
                         self.dtype))
        x = F.gelu(x)
        x = F.linear(input=x,
                     weight=self.mm_weights["model.mm_projector.2.weight"].to(
                         self.dtype),
                     bias=self.mm_weights["model.mm_projector.2.bias"].to(
                         self.dtype))

        return x.view(B, L, -1)


if __name__ == "__main__":
    model_dir = "/data/models/llava-v1.5-7b"
    model = LlavaVision(model_dir=model_dir)
    image = model.load_image("./images", num=4)

    for _ in range(10):
        x = model(image)
    print("out:\n", x[0].cpu().numpy())
    print("out shape: ", x.shape)
