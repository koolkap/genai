import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"


# -------------------------
# Core Ollama Call
# -------------------------

def llm(prompt: str, temperature: float = 0.4) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["response"].strip()


# ==========================================================
# 1. Marketing Copy Generation
# (Session 20 – Example & Mini-Lab 1)
# ==========================================================

def marketing_copy(product: str, audience: str, tone: str) -> str:
    prompt = f"""
Write a short marketing copy for the following product.

Product: {product}
Target audience: {audience}
Tone: {tone}

Constraints:
- Max 40 words
- Clear and persuasive
- Avoid exaggerated claims
"""
    return llm(prompt, temperature=0.6)


def eco_ad_copies() -> str:
    prompt = """
Write 3 short ad copies for an eco-friendly reusable water bottle.

Tone: fresh and persuasive
Max 20 words each
"""
    return llm(prompt, temperature=0.6)


# ==========================================================
# 2. Customer Support (Controlled Response Draft)
# (Derived from customer service chatbot use case)
# ==========================================================

def customer_support_reply(customer_issue: str) -> str:
    prompt = f"""
You are a customer support agent.

Customer issue:
{customer_issue}

Rules:
- Polite and empathetic
- Do not invent refunds, timelines, or guarantees
- Provide clear next steps
"""
    return llm(prompt, temperature=0.2)


# ==========================================================
# 3. Email Automation
# (Session 20 – Example & Mini-Lab 2)
# ==========================================================

def follow_up_email(context: str) -> str:
    prompt = f"""
Draft a professional follow-up email after a client meeting.

Context:
{context}

Tone: polite and concise
Include:
- Appreciation
- Next steps
"""
    return llm(prompt, temperature=0.3)


def product_launch_email(product: str, audience: str) -> str:
    prompt = f"""
Write a professional outreach email announcing our new product.

Product: {product}
Audience: {audience}
Tone: energetic and supportive
"""
    return llm(prompt, temperature=0.4)


# ==========================================================
# 4. Report Summarization
# (Session 20 – Example & Mini-Lab 3)
# ==========================================================

def report_summary(report_text: str) -> str:
    prompt = f"""
Summarize the following report into 3 business-style bullet points.

Rules:
- Preserve factual accuracy
- No assumptions or forecasts
- No additional data

Report:
{report_text}
"""
    return llm(prompt, temperature=0.1)


# ==========================================================
# Discreet Example Usage
# ==========================================================

if __name__ == "__main__":

    print("\n--- Marketing Copy ---")
    print(marketing_copy(
        product="Fitness tracking mobile app",
        audience="young professionals",
        tone="motivational"
    ))

    print("\n--- Eco Product Ads ---")
    print(eco_ad_copies())

    print("\n--- Customer Support Reply ---")
    print(customer_support_reply(
        "I was charged twice for my subscription this month."
    ))

    print("\n--- Follow-up Email ---")
    print(follow_up_email(
        "Discussed project timeline and proposal delivery by Friday."
    ))

    print("\n--- Product Launch Email ---")
    print(product_launch_email(
        product="AI-powered study app",
        audience="university students"
    ))

    print("\n--- Report Summary ---")
    print(report_summary(
        """
        Our Q2 sales increased by 15% compared to Q1.
        Marketing campaigns drove strong engagement across social media.
        However, supply chain disruptions led to delivery delays.
        """
    ))
