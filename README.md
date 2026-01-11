# Generative AI & Prompt Engineering: Full Curriculum Overview

## Overview

This repository is a comprehensive record of the core concepts, advanced techniques, and final project implementation from the Generative AI & Prompt Engineering program. It covers fundamental LLM theory, various prompting strategies, complex RAG (Retrieval-Augmented Generation) pipelines, and specialized applications in business, coding, and multimodal AI.

---

## I. Foundations and Basic Prompting (Sessions 1-4, 9, 15, 19, 20)

| Session | File Name | Key Topics Covered |
|:---|:---|:---|
| 1 | GenAI-1-what can we do with LLMs.pptx | Introduction to LLMs, Generative AI, Types of Models (GPT, Gemini, Llama, Claude), LLM Applications. |
| 2 | GenAI-2-Environment Setup and Getting Started with GPT API.pptx | Python & VS Code Setup, Virtual Environments, OpenAI API Key management, Token costs. |
| 3 | GenAI-3-Building a Chatbot with OpenAI’s API.pptx | System Role Assignment, One-Shot Prompting, Multi-turn Conversation Context (Memory), Streamlit Chatbot UI. |
| 4 | GenAI-4-AI Researcher for Summarizing Documents and Papers.pptx | Building an AI Researcher, PDF Text Extraction (PyMuPDF), Prompting for Summarization, Preprocessing. |
| 9 | GenAI-9-Zero-shot & few-shot prompting.pptx | **Zero-Shot** (simple, low-cost) vs. **Few-Shot** (consistency, complex tasks) prompting design principles. |
| 15 | Ch15. Creative AI Writing.pptx | Creative applications: Storytelling, Script Writing, Dialogue Generation, Role-Play Simulations, Prompting for style/genre. |
| 19 | Ch19. Academic Writing ^0 Paraphrasing.pptx | LLM aid in Academic Writing (Summarization, Paraphrasing, Style adaptation), Plagiarism and Hallucinated Citation Risks. |
| 20 | Ch20. Business Applications of LLMs.pptx | Use cases in Marketing (copy generation), Customer Support, Email Automation, and Report Summarization. |

---

## II. Advanced Techniques and RAG Pipelines (Sessions 5-8, 10-14, 16)

This module focuses on techniques to improve LLM reliability, coherence, and ability to handle domain-specific knowledge (RAG).

| Session | File Name | Key Topics Covered |
|:---|:---|:---|
| 5 | Ch5. RAG, LangChain, LangGraph Fundermentals.pptx | **RAG** (Retrieval-Augmented Generation) process, LangChain (Building Blocks), and LangGraph (Workflow Orchestration). |
| 6 | Ch6. RAG in practice.pptx | Local RAG Chatbot Implementation using Ollama, FAISS (Vector DB), Embeddings, and FastAPI. |
| 7 | Ch7. LangChain Simplified Demo.pptx | Practical RAG implementation using the **RetrievalQA** chain with LangChain and Ollama. |
| 8 | Ch8. LangGraph Simplified Demo.pptx | Graph-based workflow definition (Nodes and Edges) for complex RAG/Agentic pipelines. |
| 10 | GenAI-10-Chain-of-Thought & Self-Consistency.pptx | **Chain-of-Thought (CoT)** (intermediate reasoning steps) and **Self-Consistency (SC)** (aggregating multiple CoT outputs). |
| 11 | GenAI-11-ReAct Prompting.pptx | **ReAct** (Reason + Act) for interleaved reasoning and external tool use, Guardrails (stop condition, step limit). |
| 12 | GenAI-12-Iterative Refinement & Prompt Debugging.pptx | Systematic prompt improvement using the **Five Levers** (Requirement, Role, Restrictions, References, Result), Metrics & Logs. |
| 13 | Ch13. Summarization & Structured Extraction.pptx | **Extractive vs. Abstractive** summarization, converting text to structured formats (JSON, Key-Value) for automation. |
| 14 | Ch14. Handling Long Text.pptx | Mitigating token limits via **Chunking**, **Map-Reduce** summarization, and **Embeddings + Retrieval** (vector DB). |
| 16 | Ch16. Interactive Assistants (Memory Simulation).pptx | Overcoming LLM statelessness using memory simulation: **Buffer, Summary, and Windowed** memory techniques. |

---

## III. Specialized Domains (Sessions 17, 18, 21)

| Session | File Name | Key Topics Covered |
|:---|:---|:---|
| 17 | Ch17. Data Analysis in LLMs.pptx | LLM capabilities (Descriptive Stats, Schema Understanding) and limitations (Numerical fidelity) for tabular data, Prompting for tabular tasks. |
| 18 | Ch18. Code Generation & Debugging with LLMs.pptx | LLM role in generating and modifying code, common failure modes, iterative debugging workflow (error tracebacks), Security risks. |
| 21 | Ch21. Multimodal AI Theory & Practice.pptx | Multimodal AI concept (multiple inputs), **CLIP** (text-image matching), **BLIP** (image captioning), Architecture and Applications. |

---

## IV. Mini Project Implementation (Sessions 22-24)

The final project synthesizes all learned concepts into a working multimodal pipeline.

| Session | File Name | Key Deliverable / Focus |
|:---|:---|:---|
| 22 | Ch22. Mini Project Planning & Data Preparation.pptx | Define project scope, design data pipeline for multimodal tasks (text, CSV, images), prepare and preprocess datasets. |
| 23 | Ch23. Mini Project Implementation.pptx | Apply multimodal models (**BLIP, CLIP, RAG**) to the project. Implement Q&A and captioning pipeline. Key Deliverable: **Working Prototype**. |
| 24 | Ch24. Mini Project Showcase.pptx | Execute the final notebook, document the execution process (`README.md`), and write reflection (`reflection.md`) on learning and improvements. |

---

### Project Execution Files

The following files represent the critical documentation for the mini-project:

| File Name | Purpose |
|:---|:---|
|`main_project.ipynb`|The core execution code for the final multimodal RAG pipeline.|
|`README.md`|The execution guide and comprehensive curriculum overview.|
|`reflection.md`|A document for recording key learnings, challenges, and future improvements.|
|`proposal.md`|The initial project scope and dataset plan from the planning phase.|