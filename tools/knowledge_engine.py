import asyncio
import logging
from typing import List, Optional

import faiss
import numpy as np
import pymupdf
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class KnowledgeEngine:
    def __init__(self, llm_gateway, model_name: str = "ollama/llama3.2:1b"):
        self.gateway = llm_gateway
        self.model_name = model_name
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self._embedder: Optional[SentenceTransformer] = None
        # Cache for web-search results only (network calls, no token cost) — keyed by query.
        # LLM calls are intentionally NOT cached so token/cost telemetry stays accurate.
        self._web_cache: dict = {}

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder

    async def build_knowledge_base(self, file_bytes: bytes, is_pdf: bool = True):
        def _parse_and_index():
            try:
                if is_pdf:
                    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                    full_text = "\n".join([page.get_text() for page in doc])
                else:
                    full_text = file_bytes.decode('utf-8', errors='ignore')

                chunk_size = 500
                overlap = 50
                self.chunks = []

                start = 0
                text_len = len(full_text)

                while start < text_len:
                    end = min(start + chunk_size, text_len)

                    if end < text_len:
                        last_space = full_text.rfind(' ', start, end)
                        if last_space != -1 and last_space > start:
                            end = last_space

                    chunk = full_text[start:end].strip()
                    if chunk:
                        self.chunks.append(chunk)

                    start = end - overlap

                if self.chunks:
                    embedder = self._get_embedder()
                    embeddings = embedder.encode(self.chunks)
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(np.array(embeddings).astype('float32'))
                    logger.info(f"Zaindeksowano {len(self.chunks)} semantycznych fragmentów w FAISS.")
            except Exception as e:
                logger.error(f"Błąd budowy lokalnej bazy RAG: {e}")

        await asyncio.to_thread(_parse_and_index)

    def _sync_search_local_rag(self, query: str) -> str:
        if self.index is None or not self.chunks:
            return ""
        try:
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query]).astype('float32')
            distances, indices = self.index.search(query_embedding, k=3)
            results = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
            return "\n---\n".join(results)
        except Exception as e:
            logger.warning(f"Błąd FAISS: {e}")
            return ""

    async def _search_local_rag(self, query: str) -> str:
        return await asyncio.to_thread(self._sync_search_local_rag, query)

    async def _generate_query(self, chunk_text: str) -> str:
        prompt = (
            "Zamień poniższy potoczny fragment mowy na zapytanie do wyszukiwarki (max 3-5 słów kluczowych). "
            "Zwróć TYLKO wygenerowane zapytanie.\n\n"
            f"Tekst: {chunk_text}"
        )
        response = await self.gateway.execute_raw(
            prompt=prompt, model=self.model_name, temperature=0.1, max_tokens=20, agent_role="QueryGenerator"
        )
        return response.strip(' "\'\n')

    def _sync_search_web(self, query: str) -> str:
        key = query.strip().lower()
        if key in self._web_cache:
            return self._web_cache[key]
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                out = "" if not results else "\n".join(
                    [f"- {res.get('body', '')}" for res in results if res.get('body')])
        except Exception:
            out = ""
        self._web_cache[key] = out
        return out

    async def _search_web(self, query: str) -> str:
        return await asyncio.to_thread(self._sync_search_web, query)

    async def _compress_context(self, raw_data: str) -> str:
        prompt = (
            "Wyekstrahuj z poniższych wyników wyłącznie suche fakty, odrzucając szum. "
            "Zwróć wynik w formie krótkiej, wypunktowanej listy.\n\n"
            f"Dane:\n{raw_data}"
        )
        return await self.gateway.execute_raw(
            prompt=prompt, model=self.model_name, temperature=0.1, max_tokens=300, agent_role="ContextCompressor"
        )

    async def retrieve(self, chunk_text: str) -> str:
        fallback_response = "Brak wyników z zewnętrznej bazy."
        try:
            query = await self._generate_query(chunk_text)
            if not query: return fallback_response

            rag_context = await self._search_local_rag(query)
            if not rag_context:
                rag_context = await self._search_web(query)

            if not rag_context: return fallback_response

            if len(rag_context) > 500:
                compressed = await self._compress_context(rag_context)
                return compressed.strip() if compressed.strip() else fallback_response

            return rag_context.strip()
        except Exception:
            return fallback_response
