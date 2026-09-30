# RAG From Scratch — FAISS + BM25 Hybrid Search

A Retrieval-Augmented Generation pipeline built from the ground up (no LangChain retrieval chains, no black-box RAG library) to see exactly how chunking, vector search, and grounding actually work.

**Try it yourself → [Open in Colab](https://colab.research.google.com/) | Just run all cells and paste a free [Groq](https://console.groq.com) API key.**

## Problem
LLMs don't know a company's private documents, and if you just ask them anyway, they guess. I wanted to build the retrieval layer myself — chunking, embeddings, search, grounding — instead of using a pre-built RAG chain, to actually understand where each piece can fail.

## Approach
- Split a company handbook into overlapping chunks (Recursive Character Splitting)
- Indexed the chunks two ways: dense vectors (FAISS, cosine similarity) and keyword search (BM25)
- Combined both with Reciprocal Rank Fusion (RRF) hybrid search
- Reordered retrieved chunks so the best ones sit at the start/end of the prompt (fixes the "lost in the middle" problem)
- Forced the LLM to answer only from retrieved context, and say "I don't know" when the answer isn't there
- Built an 8-question test set to measure retrieval quality instead of just eyeballing answers

## Result (from my own run)
| Metric | Result |
|---|---|
| Chunks indexed | 11 |
| Best retrieval method | Dense FAISS — 100% hit-rate @2 |
| Hybrid (RRF) hit-rate @2 | 100% |
| Out-of-scope questions correctly refused | 2 / 2 |

Sample output:
> **Q: How many days of annual leave do I get and how many can I carry over?**
> A: You receive 24 days of paid annual leave each year, and you can carry over up to 5 unused days to the next year.

> **Q: What is the CEO's salary?** *(not in the document)*
> A: I don't know based on the handbook.

## Tech Stack
Python · FAISS · Sentence-Transformers · BM25 (rank_bm25) · LangChain text splitters · Groq API (`openai/gpt-oss-120b`) · pandas · matplotlib

## Why this matters
This is Project 1 of a 5-project Agentic AI series, each one built on the failure points of the last: RAG → structured/audited outputs → autonomous tool-calling agents → multimodal (vision) RAG → real-time Corrective RAG with routing and caching.

---
**Akshat Kesharwani** — Fresher Data Analyst / Data Scientist
Portfolio: https://akshatkesharwani-info.github.io/ | GitHub: https://github.com/akshatkesharwani-info
