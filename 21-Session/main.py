from pipelines.multimodal_pipeline import run_multimodal_pipeline


if __name__ == "__main__":
    result = run_multimodal_pipeline(
        image_path="assets/sample.jpg",
        text_query="eco-friendly reusable water bottle"
    )

    print("\n--- IMAGE CAPTION ---")
    print(result["image_caption"])

    print("\n--- TEXT QUERY ---")
    print(result["text_query"])

    print("\n--- MATCH RESULT ---")
    print(result["match_result"])
