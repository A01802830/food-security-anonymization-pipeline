"""
examples/merge_filter_example.py

Ejemplo básico de uso.
- Procesar por lotes y revisar manualmente la salida
- Merge + filter de una sola entrevista
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.candidate_merger import CandidateMerger
from src.ner.entity_filter import EntityFilter



# ── Procesar por lote + revisión manual ─────────────────────────────────────────────────────────────
print("\n\n─────────────────────────")
print(" Procesar por lote sin guardar + revisión manual ")
print("─────────────────────────")

mergerObject  = CandidateMerger(gliner_medium_threshold=0.75)
filterObject  = EntityFilter(filter_medium=False)

spacy_dir = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_spacy"
gliner_dir = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_gliner"

merged_docs = mergerObject.merge_directory(
	spacy_dir=spacy_dir,
	gliner_dir=gliner_dir
)

print("\nRevisar uno de los resultados...")
# muestra por entidad y qué se elimina y por qué
filterObject.preview(merged_docs[0])




# ── Procesar por archivo ─────────────────────────────────────────────────────────────
print("\n\n─────────────────────────")
print(" Procesar por archivo ")
print("─────────────────────────")

mergerObject = CandidateMerger(gliner_medium_threshold=0.75)
filterObject = EntityFilter( extra_blacklist={"temporal", "jornalero"})

spacy_file = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_spacy" / "C1_transcripts_candidates.json"
gliner_file = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_gliner" / "C1_transcripts_candidates.json"

merged_doc = mergerObject.merge_from_files(
	spacy_path=spacy_file,
	gliner_path=gliner_file
)
# devuelve MergedDocument con solo las que pasan, directamente aplicamos los cambios después de revisar
clean_doc = filterObject.filter(merged_doc)   
print(clean_doc.summary())

# Ver qué se eliminó y por qué
for r in clean_doc.removed:
	print(r["entity"].text, "→", r["reason"])
