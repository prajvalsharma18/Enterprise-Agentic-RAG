"""Factory for the chat model shared by the LangGraph and evaluation flows."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional


def extract_text_content(content: Any) -> str:
    """Extract response text from plain strings or LangChain content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
        raise TypeError("LLM response content mapping has no textual 'text' field.")

    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)

        if text_parts or not content:
            return "".join(text_parts)
        raise TypeError("LLM response content contains no supported text blocks.")

    block_text = getattr(content, "text", None)
    if isinstance(block_text, str):
        return block_text
    raise TypeError(f"Unsupported LLM response content type: {type(content).__name__}.")


def create_chat_model(
    environ: Optional[Mapping[str, str]] = None,
    temperature: float = 0.1,
):
    """Create the configured LangChain chat model without implicit fallback."""
    config = os.environ if environ is None else environ
    provider = config.get("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        api_key = config.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be configured."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=temperature,
            use_responses_api=True,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.get("OLLAMA_MODEL", "llama3.2"),
            "temperature": temperature,
        }
        base_url = config.get("OLLAMA_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    raise ValueError("LLM_PROVIDER must be either 'openai' or 'ollama'.")
