# Food Security Interviews: Hybrid Anonymization Pipeline (spaCy + GLiNER + LLM)

Automatic anonymization of Spanish interview transcripts from a food security study.
The pipeline detects sensitive named entities (people `PERSONA`, places `LUGAR` and
organizations `ORGANIZACION`) with two NER models, merges and filters their candidates,
asks a local LLM (via Ollama) to confirm or reject each one, and finally replaces every confirmed
entity in the original text with a numbered placeholder such as `[PERSONA_4]`.

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Data layout](#data-layout)
5. [Running the notebooks step by step](#running-the-notebooks-step-by-step)
6. [Results so far](#results-so-far)
7. [Next steps](#next-steps)

## Pipeline overview

```
 raw interviews (.txt)
        │
        ├──► 01 spaCy NER ─────────────┐
        │    es_core_news_lg           │
        │                              ├──► 03 compare ──► 04 merge + filter ──► 05 LLM validation ──► 06 anonymize + evaluate
        └──► 02 GLiNER NER ────────────┘                   (CandidateMerger,        (Ollama,              (EntityReplacer,
             zero-shot, 3 labels                             EntityFilter)            qwen2.5:32b)          AnonymizationEvaluator)
```

| Stage | Notebook | Input | Output |
|---|---|---|---|
| 1 | `01_spacy_ner.ipynb` | `data/raw/entrevistas_originales/*.txt` | `data/processed/entidades_candidatas_spacy/*_candidates.json` |
| 2 | `02_gliner_ner.ipynb` | `data/raw/entrevistas_originales/*.txt` | `data/processed/entidades_candidatas_gliner/*_candidates.json` |
| 3 | `03_spacy_vs_gliner.ipynb` | outputs of 1 and 2 | comparison tables (no files) |
| 4 | `04_merge_filter.ipynb` | outputs of 1 and 2 | `data/processed/entidades_candidatas_merged/*_merged.json` |
| 5 | `05_llm_refinement.ipynb` | merged candidates + original text | `data/processed/entidades_validadas/*_validated.json` + CSV summary |
| 6 | `06_anonimization_texts.ipynb` | validated entities + original text | `data/processed/entrevistas_anonimizadas/*_anonimizado.txt` |

## Repository structure

```
.
├── notebooks/
│   ├── 01_spacy_ner.ipynb
│   ├── 02_gliner_ner.ipynb
│   ├── 03_spacy_vs_gliner.ipynb
│   ├── 04_merge_filter.ipynb
│   ├── 05_llm_refinement.ipynb
│   ├── 06_anonimization_texts.ipynb
├── src/
│   ├── ner/
│   │   ├── spacy_ner.py           # SpacyNER, DocumentResult
│   │   ├── gliner_ner.py          # GlinerNER, DocumentResult
│   │   ├── candidate_merger.py    # CandidateMerger
│   │   └── entity_filter.py       # EntityFilter
│   ├── llm/
│   │   ├── snippet_builder.py     # SnippetBuilder
│   │   ├── ollama_client.py       # OllamaEntityValidator
│   │   └── schemas.py             # LLMInput, LLMOutput (pydantic)
│   ├── anonymization/
│   │   └── replacer.py            # EntityReplacer
│   └── evaluation/
│       └── metrics.py             # AnonymizationEvaluator
├── data/                          # not versioned
├── models/                        # not versioned
├── requirements.txt
├── .gitignore
└── README.md
```

All notebooks add the project root to `sys.path` (`PROJECT_ROOT = Path("../")`), so run them
from inside `notebooks/`.

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2. Virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
python -m spacy download es_core_news_lg

# 4. Ollama (only needed for stage 5)
#    Install from https://ollama.com, then:
ollama pull qwen2.5:32b          # or qwen2.5:7b / qwen2.5:14b for less memory
ollama serve                     # listens on http://localhost:11434
```

The first GLiNER run downloads `urchade/gliner_multi_pii-v1` (about 500 MB) into the Hugging Face
cache. Everything runs on CPU, and a CUDA GPU is used automatically if available.

## Data layout

Create these folders locally and place the files there (they are ignored by git):

```
data/
├── raw/
│   └── entrevistas_originales/        # C1_transcript.txt ... C59_transcript.txt
├── ground_truth/
│   └── entrevistas_anotadas/          # manually anonymized versions, entities marked as **tag**
└── processed/                         # created by the notebooks
```

The corpus has **59 interviews (about 644,000 words)**, automatically transcribed with speaker
diarization (es-MX) and not manually corrected.

## Running the notebooks step by step

### 01 · spaCy NER (`01_spacy_ner.ipynb`)
Loads every `.txt`, runs `SpacyNER(model="es_core_news_lg")` keeping only `PER`, `LOC` and `ORG`
(mapped to `PERSONA`, `LUGAR`, `ORGANIZACION`), and then:
- shows corpus statistics and `displacy` visualizations
- allows a manual review of one interview with context
- analyses entity frequencies and flags obvious false positives and possible misses
- exports one `*_candidates.json` per interview

Use `es_core_news_sm` if you run out of memory.

### 02 · GLiNER NER (`02_gliner_ner.ipynb`)
Runs `GlinerNER` with the labels `['persona', 'lugar', 'organizacion']`.

| Parameter | Value |
|---|---|
| `GLINER_MODEL` | `urchade/gliner_multi_pii-v1` (or the fine-tuned checkpoint) |
| `THRESHOLD` | `0.55` (calibrated in §3 of the notebook) |
| `chunk_size` | 300 |

Secondary labels (`pais`, `avenida`, `universidad`, `secretaria`) are mapped back to the three
main labels before exporting.

### 03 · spaCy vs GLiNER (`03_spacy_vs_gliner.ipynb`)
Loads both candidate sets and compares, per interview, which entities each model finds alone
and which both agree on. This was used to decide how to weight each source in the merge.

### 04 · Merge + filter (`04_merge_filter.ipynb`)
Final batch configuration (§3):
```python
merger = CandidateMerger(gliner_medium_threshold=0.95)
filt   = EntityFilter()                      # defaults: keeps high and medium
merged_docs = merger.merge_directory(spacy_dir=..., gliner_dir=...)
filt.filter_batch(merged_docs, OUTPUT_DIR)
```

**Confidence tiers** assigned by the merger (`src/ner/candidate_merger.py`):

| Tier | When | Merged score |
|---|---|---|
| `alta` (high) | Found by **both** spaCy and GLiNER | spaCy 1.0 + GLiNER score, about 1.5 to 2.0 |
| `media` (medium) | Found by **spaCy only**, or by **GLiNER only** with score ≥ `gliner_medium_threshold` (0.95) | 1.00 (spaCy) or 0.95 to 1.00 (GLiNER) |
| `baja` (low) | Found by **GLiNER only** with score < 0.95 | < 0.95 |

**Filter rules** (`EntityFilter`) remove `baja` candidates (and `media` too if
`filter_medium=True`), strings shorter than 3 characters, a domain blacklist, noise patterns
(numbers, punctuation, articles) and possessives without a proper noun (`mi…`, `mis…`, `sus…`).
This keeps the LLM input small. For example, C10 goes from 310 candidates (39 high, 100 medium,
171 low) to 132.

Test on a single document first (§1), try stricter settings (§2 used
`EntityFilter(filter_medium=True, extra_blacklist={"temporal", "jornalero"})`), then run the batch (§3).

### 05 · LLM refinement (`05_llm_refinement.ipynb`)
1. Check the Ollama connection (`OllamaEntityValidator.health_check()`).
2. `SnippetBuilder` attaches a context snippet from the original text to every candidate.
3. Inspect the input before sending it (§3).
4. Test with **one** document (§4), then run the full batch (§5).
5. The LLM confirms or rejects each candidate, corrects its label and proposes a canonical form.
   Responses are validated against the `LLMOutput` pydantic schema.

Settings: `qwen2.5:32b`, `temperature=0.0`. If Ollama runs on a remote machine, open an SSH
tunnel so it is reachable at `http://localhost:11434`.

### 06 · Anonymization + evaluation (`06_anonimization_texts.ipynb`)
```python
replacer = EntityReplacer(symbol="brackets")      # [PERSONA_1], [LUGAR_3], ...
replacer.anonymize_directory(validated_dir=..., originals_dir=..., output_dir=...)

evaluator = AnonymizationEvaluator(match_mode="partial")
```
Writes the `*_anonimizado.txt` files. The evaluation section (original vs. automatic vs. ground
truth) is still in progress.

## Results so far

Full run over the 59 interviews:

| Metric | Value |
|---|---|
| spaCy mentions detected | 7,389 |
| Candidates sent to the LLM (after merge + filter) | 4,203 |
| LLM decisions returned | 4,730 |
| Replacements in the final texts | 9,120 |
| Unique entities replaced | 2,549 |

## Requirements
spacy>=3.7
gliner
torch
pandas
pydantic>=2
ollama
tqdm
ipython
jupyter
After installing, download the Spanish spaCy model:
python -m spacy download es_core_news_lg

## Next steps

- Compare this pipeline against the GLiNER-only baseline using the same ground truth.
