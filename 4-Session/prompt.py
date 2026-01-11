"""
Use Case:
- Extract topic-relevant sentences from manuscripts/books
- Maintain confidentiality framing
- Output as numbered list

Populates two parameters:
- {book}
- {topic}
"""

# ----------------------------
# 1. SYSTEM MESSAGE
# ----------------------------

system_message = """
You are an expert text summarizer.
Your primary role today is to assist in distilling essential insights from a text I have personally authored.
Over the past three years, I have dedicated myself to this work, and it holds significant value.
It's important that the information provided remains confidential and is used solely for the purposes of analysis.
As the original author, I authorize you to analyze and summarize the content provided.
"""


# ----------------------------
# 2. PROMPT GENERATOR FUNCTION
# ----------------------------

def generate_prompt(book: str, topic: str) -> str:
    """
    Generates a formatted prompt for LLMs.

    Args:
        book (str): text segment from manuscript
        topic (str): chosen topic for extraction

    Returns:
        str: final completed prompt template
    """

    prompt = f"""
As the author of this manuscript, I am seeking your expertise in extracting insights related to the topic of '{topic}'.
The manuscript is a comprehensive work, and your role is to distill sentences where '{topic}' is a key element.

Here is a segment from the manuscript for review:

{book}

----

Instructions for Task Completion:
- Your output should be a numbered list, clearly formatted.
- Only include sentences where '{topic}' is a key element, not merely mentioned.
- If a sentence does not directly contribute to the understanding of '{topic}', exclude it.
- Aim for precision and relevance in your selections.

Example of Relevance:
- Correct: "Time management is crucial for productivity."
- Incorrect: "I had a great time at the concert."

With the provided segment and these instructions, extract the relevant sentences and present them as requested.
"""

    return prompt
