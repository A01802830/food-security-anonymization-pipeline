"""
src/llm/ollama_client.py

Cliente para Ollama corriendo en Google Cloud (o la nube en general)
Valida entidades candidatas usando el LLM y devuelve LLMOutput validado con Pydantic.

Arquitectura en la Nube:
  - VM con GPU (ej. A100 40GB) corriendo Ollama
  - Ollama expone la API en el puerto 11434
  - Este cliente hace requests HTTP a esa VM

Setup en la VM de GCloud (una sola vez):
  curl -fsSL https://ollama.ai/install.sh | sh
  ollama serve &
  ollama pull qwen2.5:32b      # o llama3.3:70b
  # Exponer fuera de localhost:
  OLLAMA_HOST=0.0.0.0 ollama serve

Instalación local:
  pip install dspy-ai pydantic requests
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

import dspy

from .schemas import (
    LLMInput, LLMOutput, EntityDecision,
    ValidateEntities, EntityLabel,
)
from .snippet_builder import SnippetBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración de conexión
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST  = "http://localhost:11434"   # cuando usas SSH tunnel
DEFAULT_MODEL        = "qwen2.5:7b"              # ajusta según lo que tengas en Ollama
REQUEST_TIMEOUT      = 120                        # segundos — modelos grandes son lentos
MAX_RETRIES          = 3
RETRY_DELAY          = 5                          # segundos entre reintentos
DEFAULT_BATCH_SIZE   = 5                         # entidades por llamada al LLM


# ---------------------------------------------------------------------------
# Cliente principal
# ---------------------------------------------------------------------------

class OllamaEntityValidator:
    """
    Valida entidades candidatas usando Ollama en Google Cloud.

    Uso básico:
        validator = OllamaEntityValidator(
            host  = "http://localhost:11434",   # SSH tunnel
            model = "qwen2.5:32b",
        )

        if validator.health_check(): # Verificar conexión
            result = validator.validate_entities(llm_input) # Validación de entidades
    """

    def __init__(
        self,
        host:        str = DEFAULT_OLLAMA_HOST,
        model:       str = DEFAULT_MODEL,
        timeout:     int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.0,   # 0 = determinista, mejor para extracción estructurada
        batch_size:  int = DEFAULT_BATCH_SIZE,  # entidades por llamada al LLM
    ):
        self.host        = host.rstrip("/")
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.batch_size  = batch_size

        self._dspy_lm    = self._setup_dspy()

    # ------------------------------------------------------------------
    # Setup DSPy
    # ------------------------------------------------------------------

    def _setup_dspy(self) -> dspy.LM:
        """
        Configura DSPy para usar Ollama como backend.
        DSPy maneja el prompt automáticamente según la firma ValidateEntities.
        """
        lm = dspy.LM(
            model    = f"ollama_chat/{self.model}",
            api_base = self.host,
            api_key  = "ollama",          # Ollama no necesita key real
            temperature = self.temperature,
            max_tokens  = 4096,
        )
        dspy.configure(lm=lm)
        return lm

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verifica que Ollama esté corriendo y el modelo esté disponible.
        """
        try:
            resp = requests.get(
                f"{self.host}/api/tags",
                timeout=10,
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            print(f"* Ollama conectado en {self.host}")
            print(f"   Modelos disponibles: {models}")

            model_available = any(self.model in m for m in models)
            if not model_available:
                print(f"x  Modelo '{self.model}' no encontrado.")
                print(f"   Instálalo con: ollama pull {self.model}")
                return False
            else:
                print(f"   Modelo '{self.model}' listo para usar! :)")
                return True

        except requests.ConnectionError:
            print(f"x No se puede conectar a Ollama en {self.host}")
            return False
        except Exception as e:
            print(f"x Error inesperado: {e}")
            return False

    # ------------------------------------------------------------------
    # Validación de un documento
    # ------------------------------------------------------------------

    def validate_entities_from_file(
        self,
        llm_input:  LLMInput,
        output_dir: Optional[Path] = None,
    ) -> LLMOutput:
        """
        Valida las entidades candidatas de un documento procesando en batches.

        Si hay más candidatos que self.batch_size, los divide en grupos y hace
        una llamada al LLM por grupo. Los resultados parciales se guardan en
        output_dir/<doc_id>_batch_N.json para no perder trabajo si algo falla.
        Al final fusiona todo en un único LLMOutput.

        Args:
            llm_input:  Input de un documento con sus candidatos.
            output_dir: Directorio donde guardar resultados (parciales y final).
        """
        if not llm_input.candidates:
            logger.info(f"{llm_input.doc_id}: sin candidatos, devolviendo vacío")
            return LLMOutput(doc_id=llm_input.doc_id, decisions=[])

        # ── Dividir candidatos en batches ─────────────────────────────────
        batches = self._make_batches(llm_input)
        n_batches = len(batches)

        if n_batches == 1:
            print(f"  {llm_input.n_candidates} candidatos en 1 llamada")
        else:
            print(f"  {llm_input.n_candidates} candidatos → {n_batches} batches "
                  f"de ~{self.batch_size}")

        # ── Procesar cada batch ───────────────────────────────────────────
        all_decisions: list[EntityDecision] = []

        for i, batch_input in enumerate(batches):

            # 1. Marca de tiempo inicial
            inicio = time.time()

            batch_num = i + 1
            print(f"    Batch {batch_num}/{n_batches} "
                  f"({len(batch_input.candidates)} entidades)…", end=" ", flush=True)

            try:
                batch_result = self._call_with_retry(batch_input)
                all_decisions.extend(batch_result.decisions)
                print(f"* ({len(batch_result.decisions)} decisiones)")

                # Guardar resultado parcial del batch para no perder trabajo
                if output_dir:
                    self._save_result(batch_result, output_dir,
                                      suffix=f"_batch_{batch_num:02d}")

            except Exception as e:
                logger.error(f"{llm_input.doc_id} batch {batch_num}: {e}")
                print(f"x Error — aplicando fallback")
                fallback = self._fallback_output(batch_input, str(e))
                all_decisions.extend(fallback.decisions)

                if output_dir:
                    self._save_result(fallback, output_dir,
                                      suffix=f"_batch_{batch_num:02d}_ERROR")
            
            # 2. Marca de tiempo final
            fin = time.time()

            # 3. Cálculo de tiempo total transcurrido
            tiempo_total = fin - inicio

            # 4. Conversión a minutos y segundos
            minutos = int(tiempo_total // 60)
            segundos = tiempo_total % 60

            print(f"        > Tiempo de ejecución: {minutos} minutos y {segundos:.2f} segundos\n")

        # ── Fusionar en un único LLMOutput ────────────────────────────────
        final_output = LLMOutput(
            doc_id    = llm_input.doc_id,
            decisions = all_decisions,
        )

        if output_dir:
            # Guardar resultado final consolidado (sobreescribe borradores de batch)
            self._save_result(final_output, output_dir)
        
        print(f"\n{'-'*55}")
        total_conf = final_output.to_dict()["stats"]["confirmed"]
        total_rej  = final_output.to_dict()["stats"]["rejected"]
        total_ent = final_output.to_dict()["stats"]["total"]
        print(f"Confirmadas = {total_conf}/{total_ent}")
        print(f"Rechazadas  = {total_rej}/{total_ent}")
        print(f'Total       = {total_ent} == {total_conf + total_rej} ({total_conf + total_rej == total_ent})')

        return final_output


    def validate_entities_from_directory(
        self,
        llm_inputs:    list[LLMInput],
        output_dir:    Optional[Path] = None,
        save_on_error: bool = True,
    ) -> list[LLMOutput]:
        """
        Valida todos los documentos de una lista llamando a
        validate_entities_from_file por cada uno.

        El batching interno de cada documento lo maneja validate_entities_from_file,
        así que aquí solo iteramos documentos.

        Args:
            llm_inputs:    Lista de LLMInput (uno por entrevista).
            output_dir:    Directorio de salida. Se crea una subcarpeta por documento
                           para separar los archivos de batch del resultado final.
            save_on_error: Si True, guarda resultado vacío ante error para trazabilidad.
        """
        results: list[LLMOutput] = []
        n = len(llm_inputs)

        for i, llm_input in enumerate(llm_inputs):
            print(f"\n{'='*55}")
            print(f"[{i+1}/{n}] {llm_input.doc_id} "
                  f"— {llm_input.n_candidates} candidatos")
            print(f"{'='*55}")

            # Subcarpeta por documento para aislar archivos de batch
            doc_output_dir = (output_dir / llm_input.doc_id) if output_dir else None

            try:
                result = self.validate_entities_from_file(
                    llm_input,
                    output_dir=doc_output_dir,
                )
                results.append(result)

                stats = result.to_dict()["stats"]
                print(f"  → confirmadas={stats['confirmed']}  "
                      f"rechazadas={stats['rejected']}")

            except Exception as e:
                logger.error(f"Error fatal en {llm_input.doc_id}: {e}")
                print(f"  x Error fatal: {e}")

                if save_on_error:
                    error_result = LLMOutput(doc_id=llm_input.doc_id, decisions=[])
                    results.append(error_result)
                    if doc_output_dir:
                        self._save_result(error_result, doc_output_dir,
                                          suffix="_ERROR")

        print(f"\n{'='*55}")
        print(f"Completado: {len(results)}/{n} documentos procesados")
        total_conf = sum(len(r.confirmed) for r in results)
        total_rej  = sum(len(r.rejected)  for r in results)
        print(f"Total confirmadas={total_conf}  rechazadas={total_rej}")

        return results

    # ------------------------------------------------------------------
    # Llamada al LLM con reintentos
    # -------------------------------------------------------------------

    def _call_with_retry(self, llm_input: LLMInput) -> LLMOutput:
        import json
        import re
        
        # En versiones nuevas de DSPy, TypedPredictor ya no existe. Se usa Predict.
        predictor = dspy.Predict(ValidateEntities)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"{llm_input.doc_id}: intento {attempt}/{self.max_retries}")
                result = predictor(
                    doc_id=llm_input.doc_id,
                    candidates=llm_input
                )
                
                # CASO A: DSPy fue inteligente y ya lo convirtió en un objeto LLMOutput
                if hasattr(result, 'salida') and not isinstance(result.salida, str):
                    if not getattr(result.salida, 'doc_id', None):
                        result.salida.doc_id = llm_input.doc_id
                    
                    # Si cometió el error de llamar "candidates" a las decisions
                    if hasattr(result.salida, "candidates") and not hasattr(result.salida, "decisions"):
                        result.salida.decisions = result.salida.candidates
                        
                    return result.salida

                # CASO B: DSPy falló en convertirlo y nos dio un texto puro (string)
                raw_text = str(result.salida)
                
                # Cazamos todo lo que parezca un JSON (incluso si el LLM pone texto antes o después)
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if not match:
                    raise ValueError(f"El LLM no devolvió formato JSON válido. Texto devuelto: {raw_text[:50]}...")
                
                parsed_data = json.loads(match.group(0))
                
                # 🛠️ Autocorrección: Si Qwen devolvió "candidates" en vez de "decisions"
                if "decisions" not in parsed_data and "candidates" in parsed_data:
                    parsed_data["decisions"] = parsed_data.pop("candidates")
                    
                # Forzamos el doc_id
                parsed_data["doc_id"] = llm_input.doc_id
                
                return LLMOutput(**parsed_data)

            except Exception as e:
                last_error = e
                logger.warning(f"{llm_input.doc_id} falló intento {attempt}: {e}")
                if attempt < self.max_retries:
                    import time
                    print(f"      Reintento {attempt}/{self.max_retries} en 5s…")
                    time.sleep(5)

        raise RuntimeError(f"LLM falló {self.max_retries} veces en '{llm_input.doc_id}': {last_error}")
    # ------------------------------------------------------------------
    # Parseo y validación de la respuesta
    # ------------------------------------------------------------------

    def _make_batches(self, llm_input: LLMInput) -> list[LLMInput]:
        """
        Divide los candidatos de un LLMInput en sub-LLMInputs de tamaño batch_size.
        Cada batch es un LLMInput válido independiente listo para pasar al LLM.
        """
        candidates = llm_input.candidates
        if len(candidates) <= self.batch_size:
            return [llm_input]

        batches = []
        for i in range(0, len(candidates), self.batch_size):
            chunk = candidates[i: i + self.batch_size]
            batches.append(LLMInput(
                doc_id      = llm_input.doc_id,
                candidates  = chunk,
                n_candidates = len(chunk),
            ))
        return batches

    @staticmethod
    def _fallback_output(llm_input: LLMInput, reason: str) -> LLMOutput:
        """
        Si el LLM falla, conserva todos los candidatos como confirmados
        para no perder PII por error técnico. Marcados para revisión manual.
        """
        # Format the reason and strictly truncate to 200 characters
        raw_reason = f"Sin decisión LLM ({reason}). Revisión manual requerida."
        safe_reason = raw_reason[:197] + "..." if len(raw_reason) > 200 else raw_reason

        decisions = [
            EntityDecision(
                entity_id      = c.entity_id,
                text           = c.text,
                confirmed      = True,   # conservador: ante duda, anonimizar
                final_label    = c.proposed_label,
                reason         = safe_reason,
                canonical_form = c.text,
            )
            for c in llm_input.candidates
        ]
        return LLMOutput(doc_id=llm_input.doc_id, decisions=decisions)


    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _save_result(
        result: LLMOutput,
        output_dir: Path,
        suffix: str = "",
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{result.doc_id}_validated{suffix}.json"
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        print(f"\n  * Guardado: {output_dir}")