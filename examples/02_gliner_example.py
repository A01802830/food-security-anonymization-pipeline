"""
examples/gliner_example.py

Ejemplo básico de uso.
- Como crear objeto GlinerNER
- Procesar texto e imprimir entidades
- Procesar en lote
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.gliner_ner import GlinerNER


# ── Procesar texto ─────────────────────────────────────────────────────────────
print("\n─────────────────────────")
print(" Procesar Texto ")
print("─────────────────────────")

ner = GlinerNER(model="urchade/gliner_multi_pii-v1", threshold=0.4)

result = ner.process("Mi vecina Mau vive en San Cristóbal y es mi esposa, ella tiene abuelos que no ganan tanto dinero.", doc_id="entrevista_01")
ner.print_entities(result)




# ── Procesar en lote ─────────────────────────────────────────────────────────────
print("\n\n─────────────────────────")
print(" Procesar en Lote ")
print("─────────────────────────")

ner = GlinerNER()
print(f"Información del modelo: {ner.model_info()}")

results = ner.process_batch(texts=["Mi vecina Mau vive en San Cristóbal y es mi esposa, ella tiene abuelos que no ganan tanto dinero.",
                                   "Este proyecto lo empezamos a hacer con Arlet y Nora, ellas son parte de la iniciativa El Huerto."])
for r in results:
	ner.print_entities(r)
