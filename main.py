import os
import json
import fitz  # PyMuPDF
import time
from sentence_transformers import SentenceTransformer, util

# CONFIG
INPUT_DIR = './input'
OUTPUT_DIR = './output'
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'  # ~90MB

# 1. Load persona and job
persona = {
    "role": "PhD Researcher in Computational Biology",
    "focus": "methodologies, datasets, and performance benchmarks"
}
job_to_be_done = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"

# 2. Load model
model = SentenceTransformer(MODEL_NAME)

# 3. Encode persona + job
persona_job_embedding = model.encode(persona["role"] + " " + persona["focus"] + " " + job_to_be_done)

# 4. Process PDFs
results = []
subsection_results = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith('.pdf'):
        continue

    doc = fitz.open(os.path.join(INPUT_DIR, file))
    sections = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue
        lines = text.split('\n')
        buffer = ""
        current_title = f"Page {page_num + 1}"
        for line in lines:
            if line.strip().isupper() or ':' in line or line.strip().startswith(tuple(str(i) for i in range(1, 10))):
                if buffer:
                    sections.append({
                        "document": file,
                        "page_number": page_num + 1,
                        "section_title": current_title,
                        "text": buffer.strip()
                    })
                current_title = line.strip()
                buffer = ""
            else:
                buffer += " " + line.strip()
        if buffer:
            sections.append({
                "document": file,
                "page_number": page_num + 1,
                "section_title": current_title,
                "text": buffer.strip()
            })

    if not sections:
        continue

    # 5. Embed and rank sections
    section_texts = [s["text"] for s in sections]
    section_embeddings = model.encode(section_texts, batch_size=16, convert_to_tensor=True)
    similarities = util.cos_sim(persona_job_embedding, section_embeddings)[0]

    ranked = sorted(zip(sections, similarities), key=lambda x: x[1], reverse=True)

    for rank, (sec, sim) in enumerate(ranked[:5], 1):  # Top 5
        results.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "section_title": sec["section_title"],
            "importance_rank": rank
        })

        # Sub-section analysis: pick top 2 paragraphs
        paragraphs = sec["text"].split('.')
        paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
        para_sims = util.cos_sim(persona_job_embedding, paragraph_embeddings)[0]
        top_idx = para_sims.topk(min(2, len(paragraphs))).indices.tolist()

        for idx in top_idx:
            refined_text = paragraphs[idx].strip()
            if refined_text:
                subsection_results.append({
                    "document": sec["document"],
                    "page_number": sec["page_number"],
                    "refined_text": refined_text
                })

# 6. Write output
output = {
    "metadata": {
        "input_documents": os.listdir(INPUT_DIR),
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "extracted_sections": results,
    "sub_section_analysis": subsection_results
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = f'output_{time.strftime("%Y%m%d_%H%M%S")}.json'
with open(os.path.join(OUTPUT_DIR, output_file), 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Done. Check {OUTPUT_DIR}/{output_file}")
