from vision.vision_stub import vision_model_caption
from vision.clip_matching import match_text_to_image


def run_multimodal_pipeline(image_path: str, text_query: str) -> dict:
    caption = vision_model_caption(image_path)
    match_result = match_text_to_image(caption, text_query)

    return {
        "image_caption": caption,
        "text_query": text_query,
        "match_result": match_result
    }
