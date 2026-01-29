from transformers import BlipProcessor, BlipForConditionalGeneration
from config.models import BLIP_MODEL

def generate_caption(image):
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)

    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)

    return processor.decode(output[0], skip_special_tokens=True)
