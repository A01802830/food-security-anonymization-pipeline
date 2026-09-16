"""
examples/ollama_validation_example.py

Ejemplo básico de uso.
- Conectar a ollama
- Validar conexión
- Extraer snippets de un json merged
- Validar entidades con LLM (en lotes)
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto al path para importar src/
PROJECT_ROOT = Path("../").resolve()  # ajusta si se corre desde otro lugar
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.snippet_builder import SnippetBuilder
from src.llm.ollama_client import OllamaEntityValidator



# ── LLM revisa entidades candidatas con fragmentos de texto ─────────────────────────────────────────────────────────────

validator = OllamaEntityValidator(
	host       = "http://0.0.0.0:11434",
	model      = "llama3.1:8b",
	batch_size = 3
)

input_file = PROJECT_ROOT / "data" / "processed" / "entidades_candidatas_merged" / "C4_transcripts_merged.json"
output_dir = PROJECT_ROOT / "examples"

builder = SnippetBuilder(
	snippet_window     = 120,   # chars de contexto a cada lado
	max_mentions       = 5      # máx de contextos por entidad al LLM
)

# Verificar conexión
if validator.health_check():

	# Extraer snippets de la entrevista
	builder = SnippetBuilder()
	llm_input = builder.build_from_file(input_file)
	
	# Esto es para que en la prueba no revise todos
	llm_input.candidates = llm_input.candidates[:5]
	llm_input.n_candidates = len(llm_input.candidates)

	# Creando directorio para guardar
	output_dir_transcript = output_dir / f"PRUEBA_{llm_input.doc_id}"
	output_dir_transcript.mkdir(parents=True, exist_ok=True)

	print(f"\nDirectorio: {output_dir_transcript}\n")

	# Validar un documento
	_ = validator.validate_entities_from_file(llm_input, output_dir)
