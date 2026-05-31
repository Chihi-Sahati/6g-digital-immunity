# Enhanced LLM Client with Multi-Backend Support
# Original: TelcoLLM Client for 6G Digital Immunity Framework
# Modifications: Added multi-backend fallback (OpenAI API / local model / g4f)
# Review Fix: Replaced unreliable single g4f dependency with robust fallback chain

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("llm_client")


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    Provides a common interface for different LLM providers,
    enabling easy switching between backends.

    Added abstraction layer for reproducibility.
    """

    @abstractmethod
    def generate(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 250
    ) -> Optional[str]:
        """Generate a response from the LLM.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Response content string, or None on failure.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the backend name for logging."""
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI API backend for production use.

    This is the RECOMMENDED backend for published research results
    due to its reliability and reproducibility.

    Added OpenAI backend as primary recommendation.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self._client = None

    def name(self) -> str:
        return f"OpenAI/{self.model}"

    def generate(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 250
    ) -> Optional[str]:
        try:
            from openai import OpenAI

            if self._client is None:
                if not self.api_key:
                    logger.warning("OpenAI API key not set. Skipping OpenAI backend.")
                    return None
                self._client = OpenAI(api_key=self.api_key)

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai"
            )
            return None
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None


class G4FBackend(LLMBackend):
    """g4f free GPT-4 backend (prototype/demo only).

    IMPORTANT LIMITATION (Peer Review Fix):
    The g4f library provides free access to GPT-4 via unofficial
    proxy providers. This is UNSUITABLE for published research due to:
      - Non-deterministic routing (different providers each call)
      - Rate limiting and intermittent availability
      - No version pinning (model behavior may change)
      - Ethical concerns regarding terms of service

    This backend is retained ONLY for demonstration purposes.
    For published results, use OpenAIBackend or LocalModelBackend.

    Documented limitations and added fallback chain.
    """

    def __init__(self):
        self.model = os.getenv("G4F_MODEL", "gpt-4")
        self._client = None

    def name(self) -> str:
        return f"g4f/{self.model}"

    def generate(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 250
    ) -> Optional[str]:
        try:
            from g4f.client import Client

            if self._client is None:
                self._client = Client()

            logger.debug(f"Using g4f backend (model={self.model}) - DEMO ONLY")
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except ImportError:
            logger.warning("g4f package not installed.")
            return None
        except Exception as e:
            logger.error(f"g4f API error: {e}")
            return None


class LocalModelBackend(LLMBackend):
    """Local LLM backend using HuggingFace transformers.

    Uses a locally-hosted model for fully reproducible research.
    Recommended for offline experimentation.

    Added local model backend for reproducibility.
    """

    def __init__(self):
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
        self._pipeline = None

    def name(self) -> str:
        return f"Local/{self.model_name}"

    def generate(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 250
    ) -> Optional[str]:
        try:
            from transformers import pipeline

            if self._pipeline is None:
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto",
                )
            # Format messages for the pipeline
            prompt = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in messages
            )
            result = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            return result[0]["generated_text"]

        except ImportError:
            logger.warning(
                "transformers not installed. Install with: pip install transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return None


class RuleBasedFallback(LLMBackend):
    """Deterministic rule-based fallback when no LLM is available.

    Generates simple intents based on telemetry thresholds.
    Used as last resort in the fallback chain.

    Added deterministic fallback for robustness.
    """

    def name(self) -> str:
        return "RuleBased-Fallback"

    def generate(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 250
    ) -> Optional[str]:
        try:
            # Extract telemetry from the last user message
            last_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_msg = m.get("content", "")
                    break

            # Simple rule-based response
            import re

            prb_match = re.search(r"PRB[:\s]+([\d.]+)", last_msg)
            prb = float(prb_match.group(1)) if prb_match else 50.0

            # Generate a conservative intent
            if prb > 80:
                intent = {
                    "intent_type": "CAPACITY_ADJUSTMENT",
                    "target_element": "bts-001",
                    "radio": {
                        "tx_power_dbm": 40.0,  # Conservative, below 43 dBm ceiling
                        "ho_hysteresis_db": 3.0,
                    },
                    "confidence": 0.7,
                    "reasoning": f"High congestion (PRB={prb:.1f}%), applying conservative power adjustment",
                }
            else:
                intent = {
                    "intent_type": "MONITORING",
                    "target_element": "bts-001",
                    "radio": {"tx_power_dbm": 43.0, "ho_hysteresis_db": 2.0},
                    "confidence": 0.9,
                    "reasoning": "Normal operating conditions, maintaining current configuration",
                }

            return json.dumps(intent)

        except Exception as e:
            logger.error(f"Rule-based fallback error: {e}")
            return None


class TelcoLLMClient:
    """Multi-backend LLM client with automatic fallback chain.

    The client tries backends in priority order:
      1. OpenAI API (production, reproducible)
      2. Local model (offline, reproducible)
      3. g4f (demo only, unreliable)
      4. Rule-based fallback (deterministic, always available)

    Refactored from single-backend to multi-backend
    with fallback chain. Addresses peer review concern about g4f reliability.
    """

    def __init__(self, backend_priority: Optional[list[str]] = None):
        """Initialize the multi-backend LLM client.

        Args:
            backend_priority: Ordered list of backend names to try.
                Defaults to: ["openai", "local", "g4f", "rulebased"]
                Can be overridden via LLM_BACKEND_PRIORITY env var.
        """
        if backend_priority is None:
            env_priority = os.getenv("LLM_BACKEND_PRIORITY")
            if env_priority:
                backend_priority = [b.strip() for b in env_priority.split(",")]
            else:
                backend_priority = ["openai", "local", "g4f", "rulebased"]

        self.backends = self._create_backends(backend_priority)
        self._active_backend: Optional[LLMBackend] = None

        logger.info(
            f"Initialized TelcoLLMClient with backends: "
            f"{[b.name() for b in self.backends]}"
        )

    def _create_backends(self, priority: list[str]) -> list[LLMBackend]:
        """Create backend instances from priority list."""
        factory = {
            "openai": OpenAIBackend,
            "g4f": G4FBackend,
            "local": LocalModelBackend,
            "rulebased": RuleBasedFallback,
        }

        backends = []
        for name in priority:
            if name in factory:
                backends.append(factory[name]())
            else:
                logger.warning(f"Unknown backend '{name}', skipping.")

        return backends

    def generate_intent(self, telemetry_data: dict) -> Optional[dict]:
        """Generate a NetworkIntent JSON from telemetry data.

        Tries each backend in priority order until one succeeds.

        Args:
            telemetry_data: Current RAN telemetry dictionary.

        Returns:
            Parsed NetworkIntent dictionary, or None if all backends fail.
        """
        from core.prompt_engine import TelcoPromptEngine

        messages = TelcoPromptEngine.build_prompt(telemetry_data)

        for backend in self.backends:
            try:
                logger.info(f"Trying backend: {backend.name()}")
                content = backend.generate(messages)

                if content is None:
                    logger.debug(
                        f"Backend {backend.name()} returned None, trying next."
                    )
                    continue

                logger.debug(f"Raw LLM Response from {backend.name()}: {content[:200]}")

                # Clean up potential markdown formatting
                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                intent_json = json.loads(cleaned)

                # Validate basic structure
                if "intent_type" in intent_json or "radio" in intent_json:
                    self._active_backend = backend
                    logger.info(f"Intent generated successfully by: {backend.name()}")
                    return intent_json
                else:
                    logger.warning(
                        f"Backend {backend.name()} returned invalid intent structure, "
                        f"trying next backend."
                    )

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Backend {backend.name()} returned non-JSON: {e}, trying next."
                )
                continue
            except Exception as e:
                logger.warning(f"Backend {backend.name()} error: {e}, trying next.")
                continue

        logger.error("All LLM backends failed. No intent generated.")
        return None

    @property
    def active_backend_name(self) -> Optional[str]:
        """Name of the backend that last successfully generated an intent."""
        if self._active_backend:
            return self._active_backend.name()
        return None
