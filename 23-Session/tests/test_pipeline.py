from embeddings.build_embeddings import build_vector_db
from pipeline.multimodal_pipeline import run_pipeline

db = build_vector_db("../data/text/company_policy.txt")

result = run_pipeline(
    "../data/images/office_scene.png",
    db
)

print(result)
