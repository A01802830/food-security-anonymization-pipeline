"""
examples/spacy_example.py

Ejemplo básico de uso.
- Como crear objeto SpacyNER
- Procesar texto e imprimir entidades
- Procesar en lote
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.spacy_ner import SpacyNER


# ── Procesar texto ─────────────────────────────────────────────────────────────
print("\n─────────────────────────")
print(" Procesar Texto ")
print("─────────────────────────")

ner = SpacyNER(model="es_core_news_lg", labels=['PER', 'LOC'])

result = ner.process("Mi nombre es Ana Pérez, vivo en Chiapas.", doc_id="entrevista_01")
ner.print_entities(result)




# ── Procesar en lote ─────────────────────────────────────────────────────────────
print("\n\n─────────────────────────")
print(" Procesar en Lote ")
print("─────────────────────────")

ner = SpacyNER()
print(f"Información del modelo: {ner.model_info()}")

results = ner.process_batch(texts=["Mi nombre es Ana Pérez, vivo en Chiapas.",
								  "Mi vecina Mau vive en San Cristóbal."])
for r in results:
	ner.print_entities(r)
