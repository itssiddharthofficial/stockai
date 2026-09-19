"""
Gemma 4 12B LLM Engine - local chat inference via Ollama.

Ollama is used instead of llama-cpp-python: it is already installed, needs no
MSVC/CMake toolchain on Windows, and handles weight loading and residency.
Requires: `ollama pull gemma4:12b` (done during setup).
"""
import json
import logging

import requests

from config import CHAT_CONFIG, CHAT_MODELS, GEMMA_CONFIG

logger = logging.getLogger(__name__)


class GemmaLLMEngine:
    """Wrapper around Gemma 4 12B served by a local Ollama daemon."""

    def __init__(self):
        logger.info("⏳ Connecting to Gemma 4 12B via Ollama...")
        self.model_name = GEMMA_CONFIG["model_name"]
        self.host = GEMMA_CONFIG["host"].rstrip("/")

        self._verify_model()

        logger.info("✓ Gemma 4 12B ready (%s)", self.model_name)
        logger.info("  Mode: CPU inference — expect minutes per response")

    def available(self):
        """Models actually pulled on this machine."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return {m["name"] for m in resp.json().get("models", [])}
        except Exception:
            return set()

    def resolve(self, model: str = None) -> str:
        """Pick a model name, falling back to the default if unknown."""
        if model and model in CHAT_MODELS:
            return model
        return self.model_name

    def _verify_model(self):
        """Confirm the Ollama daemon is up and the model is pulled."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error("❌ Cannot reach Ollama at %s: %s", self.host, e)
            logger.error("   Start it with: ollama serve")
            raise

        available = {m["name"] for m in resp.json().get("models", [])}
        if self.model_name not in available:
            logger.error("❌ Model %s not found. Available: %s",
                         self.model_name, sorted(available) or "none")
            logger.error("   Run: ollama pull %s", self.model_name)
            raise RuntimeError(f"Ollama model {self.model_name} not pulled")

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a raw completion from Gemma 4 12B."""
        if max_tokens is None:
            max_tokens = GEMMA_CONFIG["max_tokens"]

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    # gemma4 is a reasoning model; without this its thinking
                    # tokens consume the whole num_predict budget and the
                    # visible answer comes back empty.
                    "think": False,
                    "keep_alive": GEMMA_CONFIG["keep_alive"],
                    "options": {
                        "temperature": GEMMA_CONFIG["temperature"],
                        "top_p": GEMMA_CONFIG["top_p"],
                        "num_predict": max_tokens,
                        "num_ctx": GEMMA_CONFIG["n_ctx"],
                    },
                },
                timeout=GEMMA_CONFIG["timeout"],
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

        except requests.Timeout:
            logger.error("Generation timed out after %ss", GEMMA_CONFIG["timeout"])
            return ("Error: Gemma timed out. CPU inference on a 12B model is slow — "
                    "raise GEMMA_CONFIG['timeout'] or switch to a smaller model.")
        except Exception as e:
            logger.error("Generation error: %s", e)
            return f"Error: Could not generate response ({e})"

    def warmup(self):
        """Load the weights into RAM now, so the first real question does not
        pay the ~30s load cost on top of generation."""
        try:
            logger.info("🔥 Pre-warming %s (loading weights)...", self.model_name)
            requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": GEMMA_CONFIG["keep_alive"],
                    "options": {"num_predict": 1, "num_ctx": GEMMA_CONFIG["n_ctx"]},
                },
                timeout=GEMMA_CONFIG["timeout"],
            )
            logger.info("✓ %s resident in memory", self.model_name)
        except Exception as e:
            logger.warning("Warm-up failed (not fatal): %s", e)

    def chat_stream(self, user_message: str, system_prompt: str = None,
                    model: str = None):
        """Yield response text chunk by chunk as Gemma generates it.

        This is what makes the UI feel fast: first words appear in seconds
        instead of after the full multi-minute generation.
        """
        if system_prompt is None:
            system_prompt = CHAT_CONFIG["system_prompt"]

        try:
            with requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.resolve(model),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": True,
                    "think": False,
                    "keep_alive": GEMMA_CONFIG["keep_alive"],
                    "options": {
                        "temperature": GEMMA_CONFIG["temperature"],
                        "top_p": GEMMA_CONFIG["top_p"],
                        "num_predict": GEMMA_CONFIG["max_tokens"],
                        "num_ctx": GEMMA_CONFIG["n_ctx"],
                    },
                },
                timeout=GEMMA_CONFIG["timeout"],
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = payload.get("message", {}).get("content") or ""
                    if chunk:
                        yield chunk
                    if payload.get("done"):
                        break

        except requests.Timeout:
            logger.error("Stream timed out after %ss", GEMMA_CONFIG["timeout"])
            yield "\n\n[Error: Gemma timed out.]"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"\n\n[Error: {e}]"

    def chat(self, user_message: str, system_prompt: str = None) -> str:
        """Chat with Gemma 4 12B using Ollama's chat endpoint."""
        if system_prompt is None:
            system_prompt = CHAT_CONFIG["system_prompt"]

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": GEMMA_CONFIG["keep_alive"],
                    "options": {
                        "temperature": GEMMA_CONFIG["temperature"],
                        "top_p": GEMMA_CONFIG["top_p"],
                        "num_predict": GEMMA_CONFIG["max_tokens"],
                        "num_ctx": GEMMA_CONFIG["n_ctx"],
                    },
                },
                timeout=GEMMA_CONFIG["timeout"],
            )
            resp.raise_for_status()
            message = resp.json().get("message", {})
            content = (message.get("content") or "").strip()
            if not content:
                # Ran out of budget mid-reasoning; show the reasoning rather
                # than an empty bubble.
                content = (message.get("thinking") or "").strip()
            return content

        except requests.Timeout:
            logger.error("Chat timed out after %ss", GEMMA_CONFIG["timeout"])
            return ("Error: Gemma timed out. CPU inference on a 12B model is slow — "
                    "raise GEMMA_CONFIG['timeout'] or switch to a smaller model.")
        except Exception as e:
            logger.error("Chat error: %s", e)
            return f"Error: Could not generate response ({e})"


# Global instance
gemma_engine = None


def get_gemma_engine():
    """Get or create Gemma engine"""
    global gemma_engine
    if gemma_engine is None:
        gemma_engine = GemmaLLMEngine()
    return gemma_engine
