"""
examples/replacer_example.py

Ejemplo básico de uso.
- Anonimizar entrevistas
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymization.replacer import EntityReplacer



# ── Dada las entidades ya validadas, ahora sí anonimiza el texto completo ─────────────────────────────────────────────────────────────

replacer = EntityReplacer(symbol="brackets")

entidades_validadas = PROJECT_ROOT / "data" / "processed" / "entidades_validadas" / "C4_transcripts_validated.json"
entrevista_original = PROJECT_ROOT / "data" / "raw" / "entrevistas_originales" / "C4_transcripts.txt"
output_dir = PROJECT_ROOT / "examples"

result = replacer.anonymize_from_files(
	validated_json = entidades_validadas,
	original_txt   = entrevista_original,
	output_dir     = output_dir
)
