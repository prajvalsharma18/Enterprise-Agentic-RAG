"""RAGAS-only model configuration, separate from the application LLM client."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional

from langchain_core.embeddings import Embeddings

from embeddings.embedder import embed_query as embed_bge_query
from embeddings.embedder import embed_texts


def build_ragas_llm(environ: Optional[Mapping[str, str]] = None):
    """Build RAGAS's LangChain wrapper using OpenAI Chat Completions."""
    config = os.environ if environ is None else environ
    api_key = config.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("RAGAS evaluation requires OPENAI_API_KEY to be configured.")

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat_model = ChatOpenAI(
        model=config.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        use_responses_api=False,
    )

    class DefaultTemperatureRagasWrapper(LangchainLLMWrapper):
        """Keep RAGAS's generation temperature at the API-supported default."""

        def get_temperature(self, n: int) -> float:
            # RAGAS 0.2.15 otherwise chooses 1e-8 for n=1, which gpt-5.6-luna
            # rejects. The model accepts its default temperature value of 1.
            return 1.0

    return DefaultTemperatureRagasWrapper(chat_model)


class BgeM3EvaluationEmbeddings(Embeddings):
    """Expose the existing normalized BGE-M3 model through LangChain's interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_bge_query(text)

