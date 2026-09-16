"""
examples/evaluation_example.py

Ejemplo básico de uso.
- Comparar los textos anonimizados automáticamente vs lo anotado por los humanos

¡¡NOTA IMPORTANTE!!
A algunas entrevistas anotadas por humanos les faltan párrafos. Se puede notar en
el excel de C1 y C4 cuando se comparan las entrevistas fila x fila. Es decir, la
entrevista original y la anonimizada. Este desfase hace complicado poder hacer una
evaluación automática de las entidades que se detectaron por el pipeline y el ground truth.
Necesita mucho más refinamiento el código.

En el reporte, los únicos números confiables por el momento son:
* # NER anotadas x humano
* # NER anotadas automaticamente
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import AnonymizationEvaluator



# ── Dada las entidades ya validadas, ahora sí anonimiza el texto completo ─────────────────────────────────────────────────────────────


evaluator = AnonymizationEvaluator(match_mode="partial")

# Un documento
metrics = evaluator.evaluate_document(
    human_file    = PROJECT_ROOT / "data" / "ground_truth" / "entrevistas_anotadas" / "C4_transcripts.txt",
    system_file   = PROJECT_ROOT / "data" / "processed" / "entrevistas_anonimizadas" / "C4_transcripts_anonimizado.txt",
    original      = PROJECT_ROOT / "data"/ "raw" / "entrevistas_originales" / "C4_transcripts.txt"
)

metrics.print_report()
