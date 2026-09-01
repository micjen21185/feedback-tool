from core.llm_gateway import LLMGateway
from models.schemas import ChunkPayload


class KnowledgeGatekeeper:
    def __init__(self, gateway: LLMGateway, lightweight_model: str = "ollama/llama3.2:1b"):
        self.gateway = gateway
        self.model = lightweight_model

    async def needs_external_knowledge(self, chunk: ChunkPayload) -> bool:
        prompt = f"""
        Zadanie: Oceń, czy poniższy tekst wymaga twardej weryfikacji faktograficznej (czy zawiera liczby, daty, nazwy własne, statystyki lub definicje).
        Odpowiedz TYLKO i WYŁĄCZNIE jednym słowem: TAK lub NIE.

        Tekst: {chunk.text_data.clean_text}
        """

        response = await self.gateway.execute_raw(
            prompt=prompt,
            model=self.model,
            agent_role="Knowledge Gatekeeper",
            max_tokens=5,
            temperature=0.0
        )

        normalized = response.strip().upper()
        return normalized.startswith("TAK")
