from loaders.image_loader import load_image
from vision.blip_caption import generate_caption
from rag.query_engine import query_vector_db
from rag.rag_chain import generate_answer

def run_pipeline(image_path, vector_db):
    image = load_image(image_path)

    caption = generate_caption(image)

    context = query_vector_db(vector_db, caption)

    answer = generate_answer(context, caption)

    return {
        "caption": caption,
        "answer": answer
    }
