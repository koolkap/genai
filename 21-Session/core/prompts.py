def blip_caption_prompt(image_info: str) -> str:
    return f"""
You are an image captioning assistant.

Image information:
{image_info}

Generate a concise, factual, single-sentence caption.
"""


def clip_match_prompt(image_caption: str, text_query: str) -> str:
    return f"""
You are a multimodal matching system.

Image caption:
"{image_caption}"

Text query:
"{text_query}"

Determine whether the text matches the image.

Output:
- Match: Yes or No
- Explanation: one sentence
"""
