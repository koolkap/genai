from transformers import CLIPProcessor, CLIPModel
from config.models import CLIP_MODEL
import torch

def image_text_similarity(image, texts):
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    outputs = model(**inputs)
    return outputs.logits_per_image.softmax(dim=1)
