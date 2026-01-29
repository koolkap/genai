from embeddings.build_embeddings import build_vector_db
from pipeline.multimodal_pipeline import run_pipeline

db = build_vector_db("../session22_data_prep/data/text/company_policy.txt")

result = run_pipeline(
    "../session22_data_prep/data/images/office_scene.png",
    db
)

print(result)
