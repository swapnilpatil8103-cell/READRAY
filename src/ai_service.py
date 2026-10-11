"""AI provider abstraction.

Two providers are implemented:
  * ``anthropic``          – Claude via the official ``anthropic`` SDK (default).
  * ``openai_compatible``  – any OpenAI-compatible Chat Completions endpoint via the ``openai`` SDK
                             (OpenAI, Groq, OpenRouter, Google Gemini's compatibility endpoint, local Ollama…).

Keys are read server-side from Streamlit secrets or environment variables, or supplied by a visitor for
their own session only. Keys are never logged, and every error message is passed through ``redact_secrets``.
"""

from __future__ import annotations

import base64
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from .utils import redact_secrets

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
ANTHROPIC_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Google Gemini is served through its OpenAI-compatible endpoint.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

PROVIDERS = {"anthropic": "Anthropic (Claude)", "gemini": "Google Gemini", "openai_compatible": "OpenAI-compatible API"}


# --------------------------------------------------------------------------- errors


class AIError(Exception):
    """Base class. ``str(error)`` is user-safe (credentials redacted)."""


class AIConfigError(AIError):
    pass


class AIAuthError(AIError):
    pass


class AIRateLimitError(AIError):
    pass


class AITimeoutError(AIError):
    pass


class AIRefusalError(AIError):
    pass


class AIInputTooLargeError(AIError):
    pass


# --------------------------------------------------------------------------- config


@dataclass
class AIConfig:
    provider: str
    api_key: str = field(repr=False)
    model: str
    base_url: str | None = None
    max_output_tokens: int = 8000
    timeout_s: float = 120.0
    max_input_chars: int = 150_000
    use_fallbacks: bool = True
    source: str = "server"  # server | session

    @property
    def display_name(self) -> str:
        where = "server key" if self.source == "server" else "your session key"
        return f"{PROVIDERS.get(self.provider, self.provider)} · {self.model} · {where}"


@dataclass
class AppPolicy:
    """Deployment-level controls that protect the owner's API key on a public deployment."""

    access_code: str = field(default="", repr=False)
    max_requests_per_session: int = 40
    allow_user_keys: bool = True
    storage: str = "session"  # session | local
    data_dir: str = ".readray_data"
    max_pdf_mb: float = 50.0


def _lookup(name: str, secrets: Mapping | None, env: Mapping | None) -> str:
    for source in (secrets, env):
        if not source:
            continue
        try:
            value = source.get(name)
        except Exception:  # st.secrets raises if no secrets file exists
            value = None
        if value not in (None, ""):
            return str(value).strip()
    return ""


_PLACEHOLDER = re.compile(r"(?i)paste|your[-_ ].*key|key[-_ ]here|^x+$|^<.*>$")


def _key(name: str, secrets: Mapping | None, env: Mapping | None) -> str:
    """Like _lookup, but template placeholders such as "PASTE-YOUR-KEY-HERE" count as unset."""
    value = _lookup(name, secrets, env)
    return "" if _PLACEHOLDER.search(value) else value


def _as_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_number(value: str, default: float, cast=float):
    try:
        return cast(value) if value else default
    except ValueError:
        return default


def load_policy(secrets: Mapping | None = None, env: Mapping | None = None) -> AppPolicy:
    env = os.environ if env is None else env
    return AppPolicy(
        access_code=_lookup("READRAY_ACCESS_CODE", secrets, env),
        max_requests_per_session=int(_as_number(_lookup("READRAY_MAX_REQUESTS_PER_SESSION", secrets, env), 40, int)),
        allow_user_keys=_as_bool(_lookup("READRAY_ALLOW_USER_KEYS", secrets, env), True),
        storage="local" if _lookup("READRAY_STORAGE", secrets, env).lower() == "local" else "session",
        data_dir=_lookup("READRAY_DATA_DIR", secrets, env) or ".readray_data",
        max_pdf_mb=_as_number(_lookup("READRAY_MAX_PDF_MB", secrets, env), 50.0),
    )


def load_server_config(secrets: Mapping | None = None, env: Mapping | None = None) -> AIConfig | None:
    """Resolve the server-side AI configuration. Returns None when no key is configured."""
    env = os.environ if env is None else env
    provider = _lookup("READRAY_AI_PROVIDER", secrets, env).lower()
    anthropic_key = _key("ANTHROPIC_API_KEY", secrets, env)
    openai_key = _key("OPENAI_API_KEY", secrets, env)
    openai_base = _lookup("OPENAI_BASE_URL", secrets, env)
    gemini_key = _key("GEMINI_API_KEY", secrets, env) or _key("GOOGLE_API_KEY", secrets, env)
    if not provider:
        if anthropic_key:
            provider = "anthropic"
        elif gemini_key:
            provider = "gemini"
        elif openai_key or openai_base:
            provider = "openai_compatible"
    common = dict(
        max_output_tokens=int(_as_number(_lookup("READRAY_MAX_OUTPUT_TOKENS", secrets, env), 8000, int)),
        timeout_s=_as_number(_lookup("READRAY_TIMEOUT_SECONDS", secrets, env), 120.0),
        source="server",
    )
    if provider == "anthropic":
        if not anthropic_key:
            return None
        return AIConfig(
            "anthropic", anthropic_key, _lookup("ANTHROPIC_MODEL", secrets, env) or DEFAULT_ANTHROPIC_MODEL,
            use_fallbacks=_as_bool(_lookup("READRAY_ANTHROPIC_FALLBACKS", secrets, env), True), **common,
        )
    if provider == "gemini":
        if not gemini_key:
            return None
        return AIConfig("gemini", gemini_key, _lookup("GEMINI_MODEL", secrets, env) or DEFAULT_GEMINI_MODEL,
                        base_url=GEMINI_BASE_URL, **common)
    if provider == "openai_compatible":
        model = _lookup("OPENAI_MODEL", secrets, env)
        # Local servers such as Ollama accept any non-empty key.
        if not model or not (openai_key or openai_base):
            return None
        return AIConfig("openai_compatible", openai_key or "not-needed", model, base_url=openai_base or None, **common)
    return None


def missing_config_help() -> str:
    return (
        "No AI provider is configured, so AI explanations are switched off. Reading, search, notes, the library "
        "and the literature-review matrix all still work.\n\n"
        "To enable AI, add an `ANTHROPIC_API_KEY` or a `GEMINI_API_KEY` (or `OPENAI_API_KEY` + `OPENAI_MODEL` for "
        "another OpenAI-compatible provider) to `.streamlit/secrets.toml` locally or to the app's **Secrets** on Streamlit Community Cloud, "
        "then restart the app. You can also enter a key for this session only on the Settings page, if the "
        "deployment allows it."
    )


# --------------------------------------------------------------------------- messages


@dataclass
class ChatMessage:
    role: str  # user | assistant
    text: str
    images: list[tuple[bytes, str]] = field(default_factory=list)  # (data, media_type)


def total_input_chars(system: str, messages: list[ChatMessage]) -> int:
    return len(system) + sum(len(m.text) for m in messages)


class AIProvider(ABC):
    def __init__(self, config: AIConfig):
        self.config = config

    def _check_size(self, system: str, messages: list[ChatMessage]) -> None:
        size = total_input_chars(system, messages)
        if size > self.config.max_input_chars:
            raise AIInputTooLargeError(
                f"This request is {size:,} characters, above the {self.config.max_input_chars:,} limit. "
                "Select a shorter passage or reduce the context budget in Settings."
            )
        for m in messages:
            for data, _ in m.images:
                if len(data) > 5 * 1024 * 1024:
                    raise AIInputTooLargeError("Images sent to the AI must be under 5 MB. Crop a smaller region.")

    @abstractmethod
    def stream(self, system: str, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield response text incrementally. Raises AIError subclasses."""

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        return "".join(self.stream(system, messages))

    def _safe(self, exc: Exception) -> str:
        return redact_secrets(str(exc), [self.config.api_key])[:400]


class AnthropicProvider(AIProvider):
    def _client(self):
        import anthropic

        return anthropic.Anthropic(api_key=self.config.api_key, timeout=self.config.timeout_s, max_retries=2)

    @staticmethod
    def _to_api(messages: list[ChatMessage]) -> list[dict]:
        out = []
        for m in messages:
            if m.images:
                content = [
                    {"type": "image", "source": {"type": "base64", "media_type": mt, "data": base64.standard_b64encode(d).decode()}}
                    for d, mt in m.images
                ]
                content.append({"type": "text", "text": m.text})
                out.append({"role": m.role, "content": content})
            else:
                out.append({"role": m.role, "content": m.text})
        return out

    def stream(self, system: str, messages: list[ChatMessage]) -> Iterator[str]:
        import anthropic

        self._check_size(system, messages)
        client = self._client()
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            system=system,
            messages=self._to_api(messages),
        )
        use_fallbacks = self.config.use_fallbacks and self.config.model in ANTHROPIC_FALLBACK_MODELS
        attempts = [True, False] if use_fallbacks else [False]
        for with_fallbacks in attempts:
            extra = (
                {"extra_headers": {"anthropic-beta": FALLBACK_BETA}, "extra_body": {"fallbacks": "default"}}
                if with_fallbacks else {}
            )
            emitted = False
            try:
                with client.messages.stream(**kwargs, **extra) as stream:
                    for text in stream.text_stream:
                        emitted = True
                        yield text
                    final = stream.get_final_message()
                if final.stop_reason == "refusal":
                    raise AIRefusalError("The model declined to answer this request. Try rephrasing or selecting a different passage.")
                if final.stop_reason == "max_tokens":
                    yield "\n\n*[The response reached the output limit and was cut off.]*"
                return
            except AIError:
                raise
            except anthropic.BadRequestError as exc:
                # Retry once without the fallback beta if this account/model rejects it.
                if with_fallbacks and not emitted and "fallback" in str(exc).lower():
                    continue
                raise AIError(f"The AI provider rejected the request: {self._safe(exc)}") from None
            except anthropic.AuthenticationError:
                raise AIAuthError("The Anthropic API key was rejected. Check the key in your secrets or Settings.") from None
            except anthropic.PermissionDeniedError:
                raise AIAuthError("This API key does not have access to the selected model.") from None
            except anthropic.NotFoundError:
                raise AIConfigError(f"Model '{self.config.model}' was not found. Check ANTHROPIC_MODEL.") from None
            except anthropic.RateLimitError:
                raise AIRateLimitError("The AI provider's rate limit was reached. Wait a minute and try again.") from None
            except anthropic.APITimeoutError:
                raise AITimeoutError("The AI request timed out. Try a shorter passage or try again.") from None
            except anthropic.APIConnectionError:
                raise AIError("Could not connect to the AI provider. Check the network connection.") from None
            except anthropic.APIStatusError as exc:
                raise AIError(f"The AI provider returned an error (HTTP {exc.status_code}). Please try again.") from None


class OpenAICompatibleProvider(AIProvider):
    def _client(self):
        import openai

        return openai.OpenAI(
            api_key=self.config.api_key, base_url=self.config.base_url, timeout=self.config.timeout_s, max_retries=2
        )

    @staticmethod
    def _to_api(system: str, messages: list[ChatMessage]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.images:
                content: list[dict] = [
                    {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{base64.b64encode(d).decode()}"}}
                    for d, mt in m.images
                ]
                content.append({"type": "text", "text": m.text})
                out.append({"role": m.role, "content": content})
            else:
                out.append({"role": m.role, "content": m.text})
        return out

    def stream(self, system: str, messages: list[ChatMessage]) -> Iterator[str]:
        import openai

        self._check_size(system, messages)
        client = self._client()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=self._to_api(system, messages),
                max_tokens=self.config.max_output_tokens,
                stream=True,
            )
            finish = None
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    yield choice.delta.content
                finish = choice.finish_reason or finish
            if finish == "length":
                yield "\n\n*[The response reached the output limit and was cut off.]*"
            elif finish == "content_filter":
                raise AIRefusalError("The provider's content filter blocked this response.")
        except AIError:
            raise
        except openai.AuthenticationError:
            raise AIAuthError(f"The API key was rejected by {PROVIDERS.get(self.config.provider, 'the provider')}.") from None
        except openai.PermissionDeniedError:
            raise AIAuthError("This API key does not have access to the selected model.") from None
        except openai.NotFoundError:
            hint = "GEMINI_MODEL" if self.config.provider == "gemini" else "OPENAI_MODEL / OPENAI_BASE_URL"
            raise AIConfigError(f"Model '{self.config.model}' or the endpoint was not found. Check {hint}.") from None
        except openai.RateLimitError:
            raise AIRateLimitError("The AI provider's rate limit or quota was reached. Wait and try again.") from None
        except openai.APITimeoutError:
            raise AITimeoutError("The AI request timed out. Try a shorter passage or try again.") from None
        except openai.APIConnectionError:
            raise AIError("Could not connect to the AI provider. Check OPENAI_BASE_URL and the network.") from None
        except openai.BadRequestError as exc:
            hint = " This model may not accept images." if any(m.images for m in messages) else ""
            raise AIError(f"The AI provider rejected the request: {self._safe(exc)}{hint}") from None
        except openai.APIStatusError as exc:
            raise AIError(f"The AI provider returned an error (HTTP {exc.status_code}). Please try again.") from None


def get_provider(config: AIConfig | None) -> AIProvider:
    if config is None:
        raise AIConfigError(missing_config_help())
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider in ("openai_compatible", "gemini"):
        return OpenAICompatibleProvider(config)
    raise AIConfigError(f"Unknown AI provider '{config.provider}'.")
