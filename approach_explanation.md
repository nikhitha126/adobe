# Persona-Driven Document Intelligence — Approach Explanation

## Methodology

Our solution automatically extracts and ranks the most relevant document sections for a given persona and job-to-be-done.

**1. Input Handling:**  
We take a folder of PDFs, a persona description, and a specific task. PDFs are parsed page by page using PyMuPDF, extracting headings and paragraphs.

**2. Section Detection:**  
Basic heuristics split text into sections using line patterns (ALL CAPS, numbering, colons).

**3. Embedding & Relevance:**  
We use a compact, CPU-friendly `sentence-transformers` model (~90 MB) to embed:
- The combined persona + job
- Each section

Cosine similarity determines relevance. Sections are ranked by similarity score.

**4. Sub-Section Analysis:**  
The top sections are split into smaller paragraphs. Each paragraph is scored and the most relevant sub-sections are selected.

**5. Output:**  
The final JSON includes metadata, ranked sections with page numbers and titles, and top sub-sections. All processing stays within 1GB RAM, uses CPU only, and finishes within 60 seconds for small doc collections.

**6. Deployment:**  
The solution runs in a Docker container with no internet access. All models are pre-downloaded.

This method ensures generalizability for diverse personas, documents, and tasks.
