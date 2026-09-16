"""
src/llm/snippet_builder.py

Construye el input estructurado para el LLM a partir de los JSONs
filtrados que vienen del merger + filter.

Responsabilidades:
  1. Leer JSONs filtrados (formato: doc_id, text, entities)
  2. Agrupar menciones del mismo texto (correferencia)
  3. Extraer snippets de contexto para cada mención
  4. Devolver LLMInput validado con Pydantic
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

from .schemas import (
	EntityCandidate, EntityMention, LLMInput, EntityLabel
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SNIPPET_WINDOW   = 120    # caracteres de contexto a cada lado de la entidad
MARKER_START     = ">>>"  # marcador visual de inicio de entidad en el snippet
MARKER_END       = "<<<"  # marcador visual de fin


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class SnippetBuilder:
	"""
	Construye LLMInput a partir de un JSON filtrado.

	Uso:
		builder = SnippetBuilder()
		llm_input = builder.build_from_file(
			Path("data/processed/entidades_candidatas_merged/doc1_merged.json")
		)

	O en lote:
		inputs = builder.build_from_directory(
			Path("data/processed/entidades_candidatas_merged/")
		)
	"""

	def __init__(
		self,
		snippet_window:      int  = SNIPPET_WINDOW,
		max_mentions:        int  = 5,     # máx de menciones por entidad al LLM
	):
		self.snippet_window      = snippet_window
		self.max_mentions        = max_mentions

	# ------------------------------------------------------------------
	# Métodos
	# ------------------------------------------------------------------

	def build_from_file(self, json_path: Path) -> LLMInput:
		"""Construye LLMInput desde un JSON filtrado."""
		data = json.loads(json_path.read_text(encoding="utf-8"))
		return self.build_from_dict(data)

	def build_from_dict(self, data: dict) -> LLMInput:
		"""
		Construye LLMInput desde un dict con formato:
		{
		  "doc_id":   str,
		  "text":     str,
		  "entities": [{"text", "label", "start", "end", ...}]
		}
		"""
		doc_id   = data["doc_id"]
		text     = data["text"]
		entities = data.get("entities", [])

		# 1. Agrupar entidades por texto normalizado (misma entidad, múltiples spans)
		groups = self._group_entities(entities)

		# 2. Construir EntityCandidate para cada grupo
		candidates = []
		for group_key, group_entities in groups.items():
			candidate = self._build_candidate(doc_id, text, group_entities)
			candidates.append(candidate)

		# Ordenar por primera aparición en el texto
		candidates.sort(key=lambda c: c.mentions[0].start if c.mentions else 0)

		return LLMInput(doc_id=doc_id, candidates=candidates, n_candidates=len(candidates))

	def build_from_directory(self, input_dir: Path) -> list[LLMInput]:
		"""Procesa todos los JSONs en un directorio."""
		json_files = sorted(input_dir.glob("*.json"))
		if not json_files:
			raise FileNotFoundError(f"No se encontraron JSONs en {input_dir}")

		inputs = []
		for path in json_files:
			llm_input = self.build_from_file(path)
			print(f"  ✓ {path.name:<50} → {llm_input.n_candidates} candidatos")
			inputs.append(llm_input)

		return inputs

	# ------------------------------------------------------------------
	# Agrupación de menciones
	# ------------------------------------------------------------------

	def _group_entities(
		self, entities: list[dict]
	) -> dict[str, list[dict]]:
		"""
		Agrupa entidades por (texto_normalizado, label).
		Una misma entidad puede aparecer varias veces en el texto.
		"""
		groups: dict[str, list[dict]] = defaultdict(list)
		for ent in entities:
			key = self._normalize_text(ent["text"]) + "::" + ent.get("label", "UNKNOWN")
			groups[key].append(ent)
		return dict(groups)

	# ------------------------------------------------------------------
	# Construcción de EntityCandidate
	# ------------------------------------------------------------------

	def _build_candidate(
		self,
		doc_id:          str,
		text:            str,
		group_entities:  list[dict],
	) -> EntityCandidate:
		"""Construye un EntityCandidate para un grupo de menciones."""

		# Texto canónico: el más largo del grupo (que en realidad todos tienen el mismo texto x eso son un grupo)
		canonical_text = max(
			(e["text"] for e in group_entities),
			key=len
		)

		# Label: el más común en el grupo, o el de mayor confianza
		label_str = self._pick_label(group_entities)
		label     = self._parse_label(label_str)


		# Entity ID único y estable
		entity_id = f"{doc_id}::{self._normalize_text(canonical_text)}::{label_str}"

		# Snippets — limitados a max_mentions para no saturar el prompt
		mentions = self._extract_mentions(text, group_entities)

		return EntityCandidate(
			entity_id       = entity_id,
			text            = canonical_text,
			proposed_label  = label,
			mention_count   = len(group_entities),
			mentions        = mentions,
		)

	def _extract_mentions(
		self,
		text:     str,
		entities: list[dict],
	) -> list[EntityMention]:
		"""
		Extrae snippets de contexto para cada mención.
		Ordena por posición y limita a max_mentions.
		Prioriza menciones más informativas (snippets más largos).
		"""
		mentions = []
		seen_starts: set[int] = set()

		# Ordenar por posición
		sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))

		for ent in sorted_entities:
			start = ent.get("start", 0)
			end   = ent.get("end",   0)

			if start in seen_starts:
				continue
			seen_starts.add(start)

			snippet = self._build_snippet(text, start, end)
			mentions.append(EntityMention(
				snippet = snippet,
				start   = start,
				end     = end,
			))

		# Priorizar los snippets más ricos en contexto (más largos → más informativo)
		mentions.sort(key=lambda m: len(m.snippet), reverse=True)
		return mentions[: self.max_mentions]

	def _build_snippet(self, text: str, start: int, end: int) -> str:
		"""
		Extrae fragmento de texto con la entidad marcada visualmente.
		Limpia saltos de línea y espacios dobles.
		"""
		ctx_start = max(0, start - self.snippet_window)
		ctx_end   = min(len(text), end + self.snippet_window)

		before = text[ctx_start:start]
		entity = text[start:end]
		after  = text[end:ctx_end]

		# Limpiar whitespace sin perder contexto
		before = re.sub(r"\s+", " ", before).strip()
		entity = re.sub(r"\s+", " ", entity).strip()
		after  = re.sub(r"\s+", " ", after).strip()

		# Añadir "…" si el fragmento está en medio del texto
		prefix = "…" if ctx_start > 0 else ""
		suffix = "…" if ctx_end < len(text) else ""

		return f"{prefix}{before} {MARKER_START}{entity}{MARKER_END} {after}{suffix}"

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _normalize_text(text: str) -> str:
		return text.lower().strip()

	@staticmethod
	def _pick_label(entities: list[dict]) -> str:
		"""Etiqueta más frecuente en el grupo."""
		from collections import Counter
		labels = [e.get("label", "UNKNOWN") for e in entities]
		return Counter(labels).most_common(1)[0][0]

	@staticmethod
	def _parse_label(label_str: str) -> EntityLabel:
		mapping = {
			"PERSONA":      EntityLabel.PERSONA,
			"LUGAR":        EntityLabel.LUGAR,
			"ORGANIZACION": EntityLabel.ORGANIZACION,
			"MISC":         EntityLabel.MISC,
		}
		return mapping.get(label_str.upper(), EntityLabel.MISC)


	# ------------------------------------------------------------------
	# Serialización para debug
	# ------------------------------------------------------------------

	def to_prompt_dict(self, llm_input: LLMInput) -> dict:
		"""
		Versión simplificada del LLMInput para meter en el prompt.
		Excluye campos internos que el LLM no necesita ver.
		"""
		return {
			"doc_id": llm_input.doc_id,
			"candidates": [
				{
					"entity_id":      c.entity_id,
					"text":           c.text,
					"proposed_label": c.proposed_label.value,
					"mention_count":  c.mention_count,
					"contexts": [
						m.snippet for m in c.mentions
					],
				}
				for c in llm_input.candidates
			],
		}

	def print_summary(self, llm_input: LLMInput) -> None:
		"""Imprime resumen del input preparado para el LLM."""
		print(f"\n{'='*60}")
		print(f"Doc: {llm_input.doc_id} — {llm_input.n_candidates} candidatos")
		print(f"{'='*60}")
		for c in llm_input.candidates:
			print(
				f"  [{c.proposed_label.value:14}] "
				f"'{c.text}' ({c.mention_count}x) "
			)
			for m in c.mentions[:2]:  # solo 2 contextos en el resumen
				idx1 = m.snippet.index(">") - 20
				idx2 = m.snippet.index("<") + 20
				print(f"      …{m.snippet[idx1:idx2]}…")
