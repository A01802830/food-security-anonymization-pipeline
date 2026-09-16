"""
src/llm/schemas.py

Schemas Pydantic para estructurar el input y output del LLM.

DSPy se usa para definir la firma de la tarea (qué entra, qué sale)
y para manejar el prompt automáticamente.
Pydantic se usa para validar y parsear la respuesta JSON del LLM.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import dspy
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums compartidos
# ---------------------------------------------------------------------------

class EntityLabel(str, Enum):
	PERSONA        = "PERSONA"
	LUGAR          = "LUGAR"
	ORGANIZACION   = "ORGANIZACION"
	MISC           = "MISC"
	FALSO_POSITIVO = "FALSO_POSITIVO"   # el LLM puede asignar esto


# ---------------------------------------------------------------------------
# Modelos Pydantic — Input al LLM
# ---------------------------------------------------------------------------

class EntityMention(BaseModel):
	"""Una mención individual de la entidad en el texto (con su contexto)."""
	snippet:    str   = Field(description="Fragmento de texto con la entidad resaltada")
	start:      int   = Field(description="Índice de carácter donde empieza la entidad")
	end:        int   = Field(description="Índice de carácter donde termina la entidad")


class EntityCandidate(BaseModel):
	"""
	Una entidad candidata con todas sus menciones y contextos.
	Esto es lo que recibe el LLM para validar.
	"""
	entity_id:       str                        = Field(description="ID único: 'doc_id::texto_norm::label'")
	text:            str                        = Field(description="Texto de la entidad")
	proposed_label:  EntityLabel                = Field(description="Etiqueta propuesta por spaCy/GLiNER")
	mention_count:   int                        = Field(description="Número de veces que aparece en el documento")
	mentions:        list[EntityMention]        = Field(description="Contextos de cada mención")


class LLMInput(BaseModel):
	"""Input completo que se pasa al LLM para un documento."""
	doc_id:       str                   = Field(description="Identificador del documento")
	candidates:   list[EntityCandidate] = Field(description="Entidades candidatas a validar")
	n_candidates: int                   = Field(description="Total de candidatos")

	@model_validator(mode="after")
	def set_n_candidates(self) -> "LLMInput":
		self.n_candidates = len(self.candidates)
		return self


# ---------------------------------------------------------------------------
# Modelos Pydantic — Output del LLM
# ---------------------------------------------------------------------------

class EntityDecision(BaseModel):
	"""
	Decisión del LLM sobre una entidad candidata.
	"""
	entity_id:      str          = Field(description="Mismo entity_id del input")
	text:           str          = Field(description="Texto de la entidad (puede corregir capitalización)")
	confirmed:      bool         = Field(description="True si es PII real, False si es falso positivo")
	final_label:    EntityLabel  = Field(description="Etiqueta final (puede diferir de proposed_label)")
	reason:         str          = Field(
		description="Explicación breve (1 oración) de la decisión",
		max_length=200,
	)
	canonical_form: Optional[str] = Field(
		default=None,
		description="Forma canónica para el diccionario (ej. 'Ana García' para 'doña Ana', 'Ana')"
	)

	@field_validator("reason")
	@classmethod
	def reason_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("reason no puede estar vacío")
		return v.strip()

	@field_validator("final_label", mode="before")
	@classmethod
	def normalize_label(cls, v: str) -> str:
		"""Acepta variantes del LLM: 'persona', 'Persona', 'PERSONA'."""
		mapping = {
			"persona":        "PERSONA",
			"lugar":          "LUGAR",
			"organizacion":   "ORGANIZACION",
			"organización":   "ORGANIZACION",
			"org":            "ORGANIZACION",
			"misc":           "MISC",
			"falso_positivo": "FALSO_POSITIVO",
			"falso positivo": "FALSO_POSITIVO",
			"fp":             "FALSO_POSITIVO",
		}
		normalized = mapping.get(v.lower().strip(), v.upper())
		return normalized


class LLMOutput(BaseModel):
	"""Output completo del LLM para un documento."""
	doc_id:    str                  = Field(description="Identificador del documento")
	decisions: list[EntityDecision] = Field(description="Una decisión por cada candidato")

	# ── Propiedades de conveniencia ──────────────────────────────────────

	@property
	def confirmed(self) -> list[EntityDecision]:
		return [d for d in self.decisions if d.confirmed]

	@property
	def rejected(self) -> list[EntityDecision]:
		return [d for d in self.decisions if not d.confirmed]

	def to_dict(self) -> dict:
		return {
			"doc_id":    self.doc_id,
			"decisions": [d.model_dump() for d in self.decisions],
			"stats": {
				"total":     len(self.decisions),
				"confirmed": len(self.confirmed),
				"rejected":  len(self.rejected),
			},
		}


# ---------------------------------------------------------------------------
# DSPy — Firma de la tarea
# ---------------------------------------------------------------------------

class ValidateEntities(dspy.Signature):
	"""
	Eres un experto en protección de datos personales (PII) para entrevistas
	de investigación sobre seguridad alimentaria en México.

	Tu tarea es revisar una lista de entidades candidatas extraídas automáticamente
	de una entrevista. Para cada entidad recibes:
	  - Su texto y etiqueta propuesta
	  - Fragmentos de contexto donde aparece en el texto

	Debes decidir para cada entidad:
	  1. ¿Es realmente PII (información que identifica a una persona)?
		 - CONFIRMAR: nombres propios de personas, lugares específicos donde vive
		   alguien, organizaciones vinculadas a personas identificables
		 - RECHAZAR: palabras genéricas del dominio agrícola, roles sin nombre
		   (ej. "el productor"), palabras comunes, verbos, artículos, sustantivos no identificables

	  2. ¿Cuál es la etiqueta correcta?
		 - PERSONA: nombre o referencia a una persona específica
		 - LUGAR: lugar geográfico específico vinculado a alguien
		 - ORGANIZACION: organización o institución identificable
		 - MISC: entidad identificable que no entra en las anteriores
		 - FALSO_POSITIVO: no es PII, debe descartarse

	  3. ¿Cuál es la forma canónica? Si "doña Ana", "Ana" y "Ana García"
		 son la misma persona, la forma canónica es "Ana García".

	Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
	{
	  "doc_id": "<doc_id>",
	  "decisions": [
		{
		  "entity_id": "<entity_id>",
		  "text": "<texto corregido si aplica>",
		  "confirmed": true,
		  "final_label": "PERSONA",
		  "reason": "Nombre propio de persona identificada en múltiples contextos",
		  "canonical_form": <texto en forma canónica>
		}
		]
	}
	"""
	# Input
	doc_id:     str       = dspy.InputField(desc="ID del documento")
	candidates: LLMInput  = dspy.InputField(desc="Lista JSON de entidades candidatas con contextos")

	# Output
	salida:     LLMOutput = dspy.OutputField(
		desc="JSON válido con decisiones para cada entidad candidata. No agregues texto fuera del JSON."
	)
