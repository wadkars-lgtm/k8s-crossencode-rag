import os
from pathlib import Path
import json

BASE_FOLDER = os.environ['BASE_FOLDER']
# Define base directory where markdown files are located
base_dir = Path(f"{BASE_FOLDER}/website/content/en/docs")

# Function to extract passages from markdown files
'''
This breaks LLMs. You must chunk source docs into coherent 100–200 word passages, not raw paragraph splits.
'''
'''
def extract_passages_from_markdown(md_path, min_words=20, max_words=150):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    passages = []
    for p in paragraphs:
        words = p.split()
        if min_words < len(words) <= max_words:
            passages.append(p)
    return passages
'''


def extract_passages_from_markdown(md_path, min_words=50, max_words=200):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    paragraphs = content.split('\n\n')
    block, block_words = [], 0
    for para in paragraphs:
        words = para.strip().split()
        if not words:
            continue
        block.append(para.strip())
        block_words += len(words)
        if block_words >= min_words:
            joined = ' '.join(block)
            if min_words <= block_words <= max_words:
                yield joined
            block, block_words = [], 0

# Extract all passages
all_passages = []
for md_file in base_dir.rglob("*.md"):
    all_passages.extend(extract_passages_from_markdown(md_file))

# Save to a JSON file for later use
output_path = Path(f"{base_dir}/k8s_passages.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_passages, f, indent=2)

output_path