from PIL import Image


def extract_image_metadata(image_path: str) -> str:
    img = Image.open(image_path)
    return f"""
Image metadata:
- Format: {img.format}
- Size: {img.size[0]} x {img.size[1]}
- Color mode: {img.mode}
"""
