from core.ollama_client import llm
from core.prompts import clip_match_prompt


def match_text_to_image(image_caption: str, text_query: str) -> str:
    prompt = clip_match_prompt(image_caption, text_query)
    return llm(prompt, temperature=0.2)
