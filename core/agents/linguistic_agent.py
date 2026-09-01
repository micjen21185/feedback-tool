from core.llm_gateway import LLMGateway
from models.schemas import ChunkPayload, LectureMetadata, LinguisticOutput, TrailingLinguisticState


class LinguisticAgent:
    def __init__(self, gateway: LLMGateway, config_model: str):
        self.gateway = gateway
        self.model = config_model

    def _build_system_prompt(self, role: str, audience: str, escalation_flag: bool) -> str:
        escalation_rule = (
            "UWAGA: Flaga eskalacji jest WŁĄCZONA. Prelegent traci płynność. Bądź surowy, raportuj najdrobniejsze potknięcia."
            if escalation_flag else
            "Flaga eskalacji WYŁĄCZONA (Tryb Karencji). Ignoruj pojedyncze potknięcia, szukaj tylko stałych tendencji."
        )

        return f"""Jesteś surowym ekspertem lingwistycznym i analitykiem mowy. Twoim zadaniem jest ocena transkrypcji z tagami akustycznymi.
Twoja analiza musi uwzględniać kontekst demograficzny:
- Rola prelegenta: {role}
- Grupa docelowa (Audience): {audience}

{escalation_rule}

ZASADY INTERPRETACJI TAGÓW I REJESTRU JĘZYKOWEGO:
1. Pamiętaj, do kogo mówi prelegent. Slang i luźny język są błędem na konferencji naukowej, ale zaletą budującą relację na spotkaniu startupowym.
2. Krótkie pauzy (np. [pauza: 0.5s], [pauza: 1.2s]) na granicach zdań to naturalny oddech. Ignoruj je.
3. Długie pauzy (powyżej 2.0s) w środku myśli to anomalia i oznaka zagubienia. Należy je bezwzględnie raportować.

<BIBLIOTEKA WZORCÓW AKUSTYCZNYCH (FEW-SHOT)>
Wzorzec 1: Naturalna pauza na oddech
Input: "Zmienne w Javie [pauza: 0.8s] dzielą się na prymitywne i referencyjne."
Analiza LLM: Zignoruj. Pauza < 1.5s na granicy sensu to naturalna przerwa fizjologiczna.

Wzorzec 2: Zjawisko "Word Whisker" (Wypełniacze)
Input: "To jest eee... [pauza: 2.1s] mmm... bardzo ważne."
Analiza LLM: Raportuj anomalię: "Kaskada wypełniaczy (eee, mmm) połączona z długą pauzą (2.1s) wskazuje na silny stres lub utratę wątku."

Wzorzec 3: Korekta w locie (False Start)
Input: "Musimy użyć funkcji, to znaczy, [niewyraźne] metody statycznej."
Analiza LLM: Raportuj anomalię: "Zająknięcie i autokorekta połączona z niewyraźnym słowem. Zaburza to pewność przekazu."

Wzorzec 4: Dostosowanie do Widowni (Błąd Rejestru)
Input (Widownia - Inwestorzy): "No i ten ficzer zajebiście skaluje bazę."
Analiza LLM: Raportuj krytyczny błąd rejestru: "Niedopuszczalny wulgaryzm i slang ('ficzer', 'zajebiście') podczas formalnego spotkania z inwestorami (pitch)."
"""

    async def analyze(self, chunk: ChunkPayload, metadata: LectureMetadata) -> LinguisticOutput:
        is_escalated = getattr(chunk.trailing_linguistics, 'escalation_flag', False)
        system_prompt = self._build_system_prompt(metadata.speaker_role, metadata.target_audience, is_escalated)

        ld = chunk.linguistic_data
        fillers_str = ", ".join(f"{k}: {v}" for k, v in ld.detected_fillers.items()) or "brak"
        tendencies_str = ", ".join(f"{k}: {v}" for k, v in ld.detected_tendencies.items()) or "brak"

        user_prompt = f"""
                <DANE ILOŚCIOWE OKNA (obliczone deterministycznie)>
                WPM: {ld.chunk_wpm}
                Wypełniacze (łącznie): {ld.filler_words_count} | Rozkład: {fillers_str}
                Powtarzalne tendencje (łącznie): {ld.repeated_tendencies_count} | Rozkład: {tendencies_str}
                Znaczące pauzy: {ld.significant_pauses_count} (łącznie {ld.significant_pauses_duration_sec}s)
                Niewyraźne słowa: {ld.unclear_words_count} | Pewność transkrypcji: {ld.avg_transcription_confidence}

                <TEKST Z TAGAMI AKUSTYCZNYMI>
                {chunk.text_data.tagged_text}

                WYMÓG FORMATOWANIA:
                Wygeneruj odpowiedź rygorystycznie dopasowaną do wymaganego schematu wyjściowego (struktura JSON).
                - W liście 'anomalies' wymień wszystkie wykryte problemy (krótkie opisy).
                - W liście 'scored_anomalies' powtórz te same problemy z przypisaną wagą (severity): LOW / MEDIUM / HIGH / CRITICAL. Wulgaryzmy, rażące błędy rejestru lub całkowita utrata wątku = CRITICAL.
                - W polu 'dominant_tendencies' podsumuj dominujące zjawisko jednym, konkretnym zdaniem.
                - Opieraj oceny na powyższych danych ilościowych, nie zgaduj.
                """
        output: LinguisticOutput = await self.gateway.execute_structured(
            prompt=f"{system_prompt}\n{user_prompt}",
            schema_class=LinguisticOutput,
            model=self.model,
            agent_role="Linguistic Agent (Map Phase)"
        )

        output.chunk_id = f"chunk_{chunk.chunk_meta.index}"
        output.start_time = chunk.chunk_meta.start_time
        output.next_state = TrailingLinguisticState(
            prev_filler_count=ld.filler_words_count,
            escalation_flag=self._should_escalate(ld, output)
        )

        return output

    @staticmethod
    def _should_escalate(ld, output: LinguisticOutput) -> bool:
        # Deterministic escalation from hard metrics, OR'd with the model's anomaly count.
        hard_signal = (
                ld.filler_words_count >= 4
                or ld.significant_pauses_duration_sec >= 4.0
                or ld.unclear_words_count >= 3
                or ld.avg_transcription_confidence < 0.70
        )
        critical_signal = any(
            item.severity in ("HIGH", "CRITICAL") for item in output.scored_anomalies
        )
        return hard_signal or critical_signal or len(output.anomalies) > 2
