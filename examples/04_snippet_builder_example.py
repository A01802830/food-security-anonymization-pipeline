"""
examples/snippet_builder_example.py

Ejemplo básico de uso.
- Construir snippets que se van a pasar al LLM
"""

import sys
import json
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.snippet_builder import SnippetBuilder



# ── Revisar snippets para pasar al LLM ─────────────────────────────────────────────────────────────

json_path = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_merged" / "C4_transcripts_merged.json"

# Extraer snippets de la entrevista
builder = SnippetBuilder()
llm_input = builder.build_from_file(json_path=json_path)

# Imprimir un previo de como se verían
builder.print_summary(llm_input)

# Guardar snippets
input_dict = builder.to_prompt_dict(llm_input)

out_path = Path("PRUEBA_snippets.json")
out_path.write_text(
	json.dumps(input_dict, ensure_ascii=False, indent=4),
	encoding="utf-8",
)
