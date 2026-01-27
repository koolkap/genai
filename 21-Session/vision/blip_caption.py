from core.ollama_client import llm
from core.prompts import blip_caption_prompt


def generate_caption(image_info: str) -> str:
    prompt = blip_caption_prompt(image_info)
    return llm(prompt, temperature=0.4)
