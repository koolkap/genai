"""
Session 24 - Capstone Execution Script
Multimodal AI Assistant (Text + Image + RAG)

Run:
python main_project.py
"""

from embeddings.build_embeddings import build_vector_db
from pipeline.multimodal_pipeline import run_pipeline
from datetime import datetime

# -----------------------------
# Paths from Session 22
# -----------------------------
TEXT_PATH = "./data/text/company_policy.txt"
IMAGE_PATH = "./data/images/office_scene.png"

# -----------------------------
# Build Vector Database
# -----------------------------
print("Building vector database...")
vector_db = build_vector_db(TEXT_PATH)

# -----------------------------
# Run Multimodal Pipeline
# -----------------------------
print("Running multimodal pipeline...")
result = run_pipeline(IMAGE_PATH, vector_db)

# -----------------------------
# Print Results
# -----------------------------
print("\n--- FINAL OUTPUT ---")
print("Image Caption:", result["caption"])
print("AI Answer:", result["answer"])

# -----------------------------
# Save Execution Log
# -----------------------------
with open("execution_log.txt", "w") as f:
    f.write("Session 24 - Capstone Execution Log\n")
    f.write(f"Timestamp: {datetime.now()}\n\n")
    f.write(f"Image Caption:\n{result['caption']}\n\n")
    f.write(f"AI Answer:\n{result['answer']}\n")

print("\nExecution log saved to execution_log.txt")
