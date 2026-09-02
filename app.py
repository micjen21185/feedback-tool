import asyncio
import hashlib
import json
import pandas as pd
import streamlit as st
import zipfile
from datetime import datetime, timezone

from core.batch_runner import run_batch, scenarios_for_batch
from core.config_loader import Config
from core.llm_gateway import LLMGateway, check_ollama_models
from core.pipelines.evaluation_engine import EvaluationEngine
from core.pipelines.orchestrator import Orchestrator
from models.schemas import (
    ExperimentScenario, LectureMetadata, SystemConfiguration,
    AgentModelsConfig, ChunkPayload, SlideSummary, TimelinePayload,
    BatchExport
)
from observabilty.metrics_engine import ObservabilityManager

st.set_page_config(page_title="FeedbackAssistantTool", layout="wide")
st.title("🎙️ FeedbackAssistantTool - Orchestrator UI")

if 'zip_data' not in st.session_state:
    st.session_state.zip_data = {
        "is_valid": False,
        "metadata": {"speaker_role": "", "target_audience": "", "main_topic": "", "knowledge_level": "Podstawowy",
                     "strategy": ""},
        "raw_text": "",
        "formatted_text": "",
        "chunks": []
    }

# Cache of per-scenario FinalReports (kept across runs) so scenarios can be compared/evaluated.
if 'evaluated_reports' not in st.session_state:
    st.session_state.evaluated_reports = {}

# Which cached scenario's report to show by default in the persistent report view.
if 'active_report_scenario' not in st.session_state:
    st.session_state.active_report_scenario = None

# Multi-run store (composite-keyed): every RunResult, so the SAME scenario with DIFFERENT models
# is kept separately for comparison/export. Separate from the single-run 'evaluated_reports' view.
if 'runs' not in st.session_state:
    st.session_state.runs = []


def process_uploaded_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            file_list = z.namelist()

            metadata_path = next((f for f in file_list if f.endswith('metadata.json')), None)
            raw_text_path = next((f for f in file_list if f.endswith('full_raw_text.txt')), None)
            formatted_text_path = next((f for f in file_list if f.endswith('full_formatted_text.txt')), None)
            timeline_path = next((f for f in file_list if f.endswith('timeline.json')), None)

            if not metadata_path:
                st.error("❌ Błąd: Paczka nie zawiera pliku metadata.json!")
                return False

            metadata_content = json.loads(z.read(metadata_path).decode('utf-8'))
            raw_text = z.read(raw_text_path).decode('utf-8') if raw_text_path else ""
            formatted_text = z.read(formatted_text_path).decode('utf-8') if formatted_text_path else ""
            timeline = json.loads(z.read(timeline_path).decode('utf-8')) if timeline_path else {}

            chunks_data = []
            slide_summaries = {}

            for f in file_list:
                if f.endswith('.json'):
                    file_name = f.split('/')[-1] if '/' in f else f

                    if 'chunk_' in file_name:
                        chunk_json = json.loads(z.read(f).decode('utf-8'))
                        chunks_data.append(chunk_json)
                    elif 'slide_summary' in file_name:
                        folder_name = f.split('/')[0] if '/' in f else 'global'
                        slide_summaries[folder_name] = json.loads(z.read(f).decode('utf-8'))

            chunks_data.sort(key=lambda x: x.get("chunk_meta", {}).get("start_time", 0.0))

            st.session_state.zip_data = {
                "is_valid": True,
                "metadata": metadata_content,
                "raw_text": raw_text,
                "formatted_text": formatted_text,
                "timeline": timeline,
                "chunks": chunks_data,
                "slide_summaries": slide_summaries
            }

            st.success(
                f"✅ Wczytano: Metadane, Teksty, Timeline, {len(chunks_data)} chunków i {len(slide_summaries)} podsumowań slajdów.")
            return True
    except Exception as e:
        st.error(f"❌ Błąd przetwarzania paczki ZIP: {e}")
        return False


def _load_runs_from_disk() -> int:
    """Load previously-saved RunResult JSON files from RUNS_DIR into session state.
    Recovers a crashed batch's completed scenarios. Returns how many NEW runs were added
    (skips run_ids already present)."""
    import os
    from models.schemas import RunResult
    existing_ids = {r.run_id for r in st.session_state.runs}
    added = 0
    directory = Config.RUNS_DIR
    if not os.path.isdir(directory):
        return 0
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, fname), encoding="utf-8") as fh:
                r = RunResult.model_validate_json(fh.read())
            if r.run_id not in existing_ids:
                st.session_state.runs.append(r)
                existing_ids.add(r.run_id)
                added += 1
        except Exception:
            continue  # skip corrupt/partial files
    return added


def _judge_config_controls(key_prefix: str) -> dict:
    """Render judge-grounding controls in an expander; return kwargs for EvaluationEngine.
    Shared by the batch section and the manual comparison section."""
    with st.expander("⚙️ Konfiguracja sędziego (osadzenie / groundedness)"):
        excerpt_chars = st.slider(
            "Rozmiar fragmentu transkrypcji dla sędziego (znaki)", 0, 30000,
            Config.JUDGE_EXCERPT_CHARS, step=1000, key=f"{key_prefix}_excerpt_chars",
            help="0 = wyłącz wieloregionowy fragment. Więcej = lepsze osadzenie, ale drożej."
        )
        excerpt_regions = st.slider(
            "Liczba regionów (początek/środek/koniec…)", 1, 5, Config.JUDGE_EXCERPT_REGIONS,
            key=f"{key_prefix}_excerpt_regions"
        )
        probe_timestamps = st.slider(
            "Liczba sond czasowych [MM:SS] do sprawdzenia", 0, 20, Config.JUDGE_PROBE_TIMESTAMPS,
            key=f"{key_prefix}_probe_ts",
            help="0 = wyłącz sondowanie przy znacznikach czasu."
        )
        probe_window = st.slider(
            "Okno sondy czasowej (znaki wokół znacznika)", 100, 2000, Config.JUDGE_PROBE_WINDOW_CHARS,
            step=100, key=f"{key_prefix}_probe_win"
        )
        focus = st.text_area(
            "Szczególny nacisk dla sędziego (opcjonalnie)", value="",
            key=f"{key_prefix}_focus",
            help="Np. 'Zwróć szczególną uwagę na poprawność nazwisk i dat' albo 'Oceń surowo ton'."
        )
    return {
        "excerpt_chars": excerpt_chars,
        "excerpt_regions": excerpt_regions,
        "probe_timestamps": probe_timestamps,
        "probe_window_chars": probe_window,
        "focus_instruction": focus,
    }


def _batch_markdown(export) -> str:
    """Human-readable Markdown summary of a batch: header + per-run scores/cost + judge table."""
    lines = [
        f"# Raport wsadowy — {export.source_label}",
        f"Utworzono: {export.created_at}",
        "",
        "## Uruchomienia",
    ]
    for r in export.runs:
        sc = r.report.scorecard
        overall = f"{sc.overall_score}/100" if sc else "—"
        cost = r.report.telemetry.total_cost_usd
        toks = r.report.telemetry.total_tokens_in + r.report.telemetry.total_tokens_out
        lines.append(
            f"- **{r.scenario_name}** | Hegemon=`{r.models.hegemon_model}`, "
            f"Merytoryczny=`{r.models.factual_model}`, Językowy=`{r.models.linguistic_model}` | "
            f"Ocena: {overall} | Koszt: ${cost:.4f} | Tokeny: {toks}"
        )
    if export.evaluation and export.evaluation.per_scenario:
        lines += ["", "## Ocena sędziego (jakość vs. koszt)", "",
                  "| Scenariusz | Jakość/50 | Osadzenie | Koszt $ | Koszt/pkt | Lost-in-middle |",
                  "|---|---|---|---|---|---|"]
        for se in export.evaluation.per_scenario:
            cpq = round(se.total_cost_usd / se.rubric_total, 6) if se.rubric_total else "—"
            lines.append(
                f"| {se.scenario_name} | {se.rubric_total} | {se.rubric.groundedness}/10 | "
                f"{round(se.total_cost_usd, 5)} | {cpq} | {'TAK' if se.lost_in_middle_flag else 'nie'} |"
            )
        if export.evaluation.summary:
            lines += ["", export.evaluation.summary]
    return "\n".join(lines)


def _fmt_score(value) -> str:
    return f"{value}/100" if value is not None else "nie dotyczy"


def render_report(report):
    """Render a FinalReport. Called from the persistent view so it survives Streamlit reruns."""
    if report.scorecard is not None:
        sc = report.scorecard
        st.subheader(f"🏁 Ocena łączna: {sc.overall_score}/100 — {sc.readiness_verdict}")
        sm1, sm2, sm3 = st.columns(3)
        # Sub-scores are only meaningful for the swarm/two-phase pipelines. The naked monolith
        # produces a single overall score, so factual/linguistic are None → "nie dotyczy".
        sm1.metric("Merytoryka", _fmt_score(sc.factual_score))
        sm2.metric("Język", _fmt_score(sc.linguistic_score))
        if sc.slide_coverage_score is not None:
            sm3.metric("Pokrycie slajdów", _fmt_score(sc.slide_coverage_score))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧠 Analiza i Detale", "💡 Feedback", "📊 Raport Kosztowy z Roju (Telemetry)", "🔬 Debug (surowe dane)"])

    with tab1:
        st.write("### Podsumowanie Merytoryczne")
        st.info(report.analysis.factual_summary)
        st.write("### Analiza Językowa")
        st.info(report.analysis.linguistic_summary)
        if report.analysis.missed_context:
            st.warning(f"**Pominięto wątki:** {', '.join(report.analysis.missed_context)}")

        if report.analysis.unverified_claims:
            st.warning(
                "**⚠️ Twierdzenia wymagające weryfikacji** (nie potwierdzone w źródłach — "
                "NIE oznaczają błędu, ale prelegent/sędzia powinien je sprawdzić):\n\n"
                + "\n".join(f"- {c}" for c in report.analysis.unverified_claims)
            )

        if report.analysis.presentation_flow is not None:
            flow = report.analysis.presentation_flow
            st.write("### 🖼️ Przepływ Prezentacji")
            st.caption(flow.flow_summary)
            if report.analysis.slide_coverage:
                for cov in report.analysis.slide_coverage:
                    header = f"Slajd {cov.slide_id} — czas {cov.time_on_slide_sec}s [{cov.dwell_verdict}]"
                    if cov.returned_later:
                        header += " ↩ powrót"
                    with st.expander(header):
                        if cov.covered_points:
                            st.markdown("**Omówione:** " + "; ".join(cov.covered_points))
                        if cov.missed_points:
                            st.markdown("**Pominięte:** " + "; ".join(cov.missed_points))

    with tab2:
        # The full multi-paragraph mentoring essay generated by the Hegemon / monolith.
        st.write("### 📝 Rozbudowany Feedback Mentorski")
        if report.feedback.executive_summary_markdown.strip():
            st.markdown(report.feedback.executive_summary_markdown)
        else:
            st.info("Model nie wygenerował rozbudowanego eseju mentorskiego dla tego przebiegu.")
        st.divider()

        c1, c2 = st.columns(2)
        c1.write("### 💪 Mocne Strony")
        for s in report.feedback.strengths:
            c1.markdown(f"- {s}")
        c2.write("### 🛠 Obszary do poprawy")
        for a in report.feedback.areas_for_improvement:
            c2.markdown(f"- {a}")

        st.write("### 🎯 Wskazówki (Actionable Tips)")
        for t in report.feedback.actionable_tips:
            st.markdown(f"👉 {t}")
        if report.feedback.overall_message:
            st.success(f"**Główne Przesłanie:** {report.feedback.overall_message}")

    with tab3:
        st.write("### 🧮 Główne Metryki")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sumaryczny Koszt (USD)", f"${report.telemetry.total_cost_usd:.4f}")
        m2.metric("Suma Tokenów", f"{report.telemetry.total_tokens_in + report.telemetry.total_tokens_out}")
        m3.metric("Fazy Map", f"{report.telemetry.map_phases_count}")
        m4.metric("Fazy Reduce", f"{report.telemetry.reduce_phases_count}")

        # Non-penalizing substance density (swarm only): how much of the talk carried checkable
        # factual content. Informational — NOT part of the score.
        if report.total_windows:
            pct = round(100 * report.substantive_windows / report.total_windows)
            st.caption(
                f"📊 Gęstość merytoryczna (obserwacyjna, nie wpływa na ocenę): "
                f"{report.substantive_windows}/{report.total_windows} okien z treścią merytoryczną ({pct}%)."
            )

        st.write("---")
        st.write("### 🔍 Szczegóły Rozbicia (Koszt per Agent/Model)")
        if report.telemetry.phase_details:
            df = pd.DataFrame([detail.model_dump() for detail in report.telemetry.phase_details])
            st.dataframe(
                df,
                column_config={
                    "agent_role": "Rola / Agent",
                    "model_name": "Użyty Model",
                    "tokens_in": st.column_config.NumberColumn("Tokeny IN"),
                    "tokens_out": st.column_config.NumberColumn("Tokeny OUT"),
                    "cost_usd": st.column_config.NumberColumn("Koszt ($)", format="$%.5f"),
                    "time_s": st.column_config.NumberColumn("Czas (s)", format="%.2f s"),
                    "ttft_ms": st.column_config.NumberColumn("TTFT (ms)", format="%d ms")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Brak szczegółowych wpisów telemetrycznych z Roju.")

    with tab4:
        # Requests #1 & #2: see what the reducer/monolith actually produced, and the aggregated
        # data that was fed into it (for the swarm, this is the reduced map output).
        st.write("### 📥 Dane wejściowe do reduktora (Hegemon / Monolit)")
        st.caption(
            "To co model dostał na wejściu. Dla Roju (Swarm) to zagregowany i zredukowany wynik "
            "wszystkich chunków z fazy map. Dla monolitu — pełny prompt z transkrypcją."
        )
        if report.reducer_input:
            st.text_area("reducer_input", report.reducer_input, height=300, key="dbg_reducer_input")
        else:
            st.info("Brak zapisanego wejścia reduktora.")

        st.write("### 📤 Surowa odpowiedź modelu")
        st.caption(
            "Dokładnie to, co model wygenerował — zanim spróbowaliśmy wyciągnąć znaczniki. "
            "Jeśli analiza/feedback są puste, tutaj zobaczysz dlaczego (np. model nie użył tagów)."
        )
        if report.raw_reducer_response:
            st.text_area("raw_reducer_response", report.raw_reducer_response, height=400,
                         key="dbg_raw_response")
        else:
            st.info("Brak zapisanej surowej odpowiedzi.")


st.sidebar.header("⚙️ Konfiguracja Systemu")
all_models = Config.get_all_models()

scenario_choice = st.sidebar.selectbox(
    "Wybierz Scenariusz Badawczy:",
    options=[e for e in ExperimentScenario],
    format_func=lambda x: f"{x.value} - {x.name}"
)

st.sidebar.subheader("Opcje Eksperymentalne")
use_llmlingua_switch = st.sidebar.checkbox(
    "Użyj LLMLingua (Kompresja Entropijna przed Hegemonem)",
    value=False,
    help="Wymaga zainstalowanego środowiska PyTorch. Drastycznie tnie zużycie tokenów wyjściowych."
)

st.sidebar.subheader("Modele dla ról")
# Defaults = "balanced local" combo: big Hegemon for the single reduce call, mid-size
# grounded factual mapper (Polish Bielik), fast linguistic mapper. Falls back to index 0
# (first commercial model) if the preferred local model isn't in the list.
_HEGEMON_DEFAULT = "ollama/llama3.1:70b"
_FACTUAL_DEFAULT = "ollama/Speakleash/bielik-11b-v3.0-instruct:Q5_K_M"
_LINGUISTIC_DEFAULT = "ollama/llama3.1:8b"

hegemon_model = st.sidebar.selectbox(
    "Hegemon (Reduce / Monolith):",
    options=all_models,
    index=all_models.index(_HEGEMON_DEFAULT) if _HEGEMON_DEFAULT in all_models else 0
)

factual_model = st.sidebar.selectbox(
    "Agent Merytoryczny (Map):",
    options=all_models,
    index=all_models.index(_FACTUAL_DEFAULT) if _FACTUAL_DEFAULT in all_models else 0
)

linguistic_model = st.sidebar.selectbox(
    "Agent Językowy (Map):",
    options=all_models,
    index=all_models.index(_LINGUISTIC_DEFAULT) if _LINGUISTIC_DEFAULT in all_models else 0
)

st.subheader("📂 Krok 1: Wczytaj dane wejściowe")

uploaded_zip = st.file_uploader("1. Załaduj wygenerowaną paczkę ZIP z danymi (Transkrypcja + Slajdy)", type="zip")

if uploaded_zip is not None:
    if st.button("Sprawdź i załaduj strukturę paczki"):
        process_uploaded_zip(uploaded_zip)

uploaded_kb_pdf = st.file_uploader(
    "2. (Opcjonalnie) Załaduj bazę wiedzy PDF dla RAG (scenariusze 4 i 5)",
    type="pdf"
)

st.divider()

st.subheader("📋 Krok 2: Weryfikacja Metadanych Wykładu")
st.info(
    "Dane zaczytane z pliku metadata.json. Metryki ilościowe (czas, słowa, pauzy) przekazywane są do modeli automatycznie w tle.")

md = st.session_state.zip_data["metadata"]

with st.form("metadata_form"):
    col1, col2 = st.columns(2)
    with col1:
        speaker_role = st.text_input("Rola prelegenta", value=md.get("speaker_role", ""))
        target_audience = st.text_input("Grupa docelowa", value=md.get("target_audience", ""))

        kl_options = ["Brak", "Podstawowy", "Średni", "Zaawansowany", "Ekspert"]
        current_kl = md.get("knowledge_level", "Podstawowy")
        kl_index = kl_options.index(current_kl) if current_kl in kl_options else 1
        knowledge_level = st.selectbox("Poziom wiedzy odbiorców", kl_options, index=kl_index)

    with col2:
        main_topic = st.text_input("Główny temat", value=md.get("main_topic", ""))
        st.markdown("**Statystyki nagrania (Dołączone do kontekstu LLM):**")
        st.caption(
            f"Czas: **{md.get('total_duration_sec', 0)}s** | "
            f"Słowa: **{md.get('total_words', 0)}** | "
            f"WPM (Min/Max): **{md.get('slowest_chunk_wpm', 0)} / {md.get('fastest_chunk_wpm', 0)}**\n\n"
            f"Wypełniacze: **{md.get('total_filler_words', 0)}** | "
            f"Pauzy: **{md.get('total_significant_pauses', 0)} ({md.get('total_significant_pauses_duration_sec', 0)}s)** | "
            f"Złe słowa: **{md.get('total_unclear_words', 0)}**"
        )

    submit_disabled = not st.session_state.zip_data["is_valid"]
    submitted = st.form_submit_button("🚀 Uruchom Ewaluację Przemówienia", disabled=submit_disabled)

if submitted and st.session_state.zip_data["is_valid"]:
    metadata = LectureMetadata(
        speaker_role=speaker_role,
        target_audience=target_audience,
        main_topic=main_topic,
        knowledge_level=knowledge_level,
        has_knowledge_base_file=md.get("has_knowledge_base_file", False),
        total_duration_sec=md.get("total_duration_sec", 0.0),
        total_words=md.get("total_words", 0),
        fastest_chunk_wpm=md.get("fastest_chunk_wpm", 0),
        slowest_chunk_wpm=md.get("slowest_chunk_wpm", 0),
        total_filler_words=md.get("total_filler_words", 0),
        total_repeated_tendencies=md.get("total_repeated_tendencies", 0),
        total_significant_pauses=md.get("total_significant_pauses", 0),
        total_significant_pauses_duration_sec=md.get("total_significant_pauses_duration_sec", 0.0),
        total_unclear_words=md.get("total_unclear_words", 0),
        overall_transcription_confidence=md.get("overall_transcription_confidence", 0.0)
    )

    system_config = SystemConfiguration(
        scenario=scenario_choice,
        hegemon_model=hegemon_model,
        agent_models=AgentModelsConfig(
            factual_model=factual_model,
            linguistic_model=linguistic_model
        ),
        use_tools=True if scenario_choice in [ExperimentScenario.SWARM_NAIVE_RAG_WEB,
                                              ExperimentScenario.SWARM_PRESENTATION_RAG_WEB] else False,
        use_llmlingua=use_llmlingua_switch
    )
    raw_text = st.session_state.zip_data["raw_text"]
    formatted_text = st.session_state.zip_data["formatted_text"]
    chunks = st.session_state.zip_data["chunks"]
    timeline = st.session_state.zip_data["timeline"]
    slide_summaries = st.session_state.zip_data["slide_summaries"]

    parsed_chunks = [ChunkPayload(**c) for c in chunks]
    parsed_summaries = {k: SlideSummary(**v) for k, v in slide_summaries.items()}
    parsed_timeline = TimelinePayload(**timeline) if timeline else None

    with st.status(f"Orkiestrator pracuje (Scenariusz: {scenario_choice.name})...", expanded=True) as status:
        obs_manager = ObservabilityManager()
        real_gateway = LLMGateway(obs_manager)


        # Live step log ("thinking process"): each pipeline milestone is written into the status box.
        def _progress(msg: str):
            status.write(msg)


        orchestrator = Orchestrator(system_config, gateway=real_gateway, progress_cb=_progress)

        # Preflight: fail fast (with a clear message) if a selected Ollama model isn't on the
        # server — otherwise the run hangs/errors mid-pipeline on "model not found".
        selected_models = [system_config.hegemon_model, factual_model, linguistic_model, Config.UTILITY_MODEL]
        missing_models = check_ollama_models(selected_models)
        if missing_models:
            status.update(label="Brakuje modeli w Ollama", state="error")
            st.error(
                "❌ Wybrane modele nie są dostępne na serwerze Ollama:\n\n"
                + "\n".join(f"- `{m}`" for m in missing_models)
                + "\n\nPobierz je (`ollama pull <nazwa>`) albo wybierz w panelu bocznym modele, "
                  "które faktycznie masz. Sprawdź dokładne nazwy: `ollama list`."
            )
            st.stop()

        knowledge_base_bytes = uploaded_kb_pdf.getvalue() if uploaded_kb_pdf is not None else None

        try:
            report = orchestrator.execute_pipeline(
                metadata=metadata,
                raw_text=raw_text,
                formatted_text=formatted_text,
                chunks=parsed_chunks,
                timeline=parsed_timeline,
                slide_summaries=parsed_summaries,
                knowledge_base_bytes=knowledge_base_bytes
            )
        except Exception as e:
            msg = str(e).lower()
            is_oom = any(m in msg for m in (
                "process has terminated", 'signal "killed"', "signal: killed",
                "out of memory", "cudamalloc", "failed to allocate"
            ))
            status.update(label="Analiza nie powiodła się", state="error")
            if is_oom:
                st.error(
                    "❌ Serwer Ollama został zabity (prawdopodobnie brak pamięci — OOM).\n\n"
                    f"Model **{system_config.hegemon_model}** nie zmieścił się w pamięci GPU/RAM "
                    "dla tak długiego nagrania. Na karcie **NVIDIA L4 (24 GB)** model 70B się nie mieści.\n\n"
                    "**Co zrobić:**\n"
                    "- Wybierz mniejszy model Hegemona (np. `ollama/gemma3:27b`), lub\n"
                    "- Użyj scenariusza **Roju (Swarm)** zamiast monolitu — dzieli tekst na fragmenty "
                    "i zużywa znacznie mniej pamięci na długich wystąpieniach, lub\n"
                    "- Użyj modelu chmurowego (gpt-4o / Claude), który nie podlega lokalnemu OOM."
                )
            else:
                st.error(f"❌ Analiza nie powiodła się: {e}")
            st.stop()

        status.update(label="Analiza zakończona sukcesem!", state="complete")

    st.success("Analiza zakończona sukcesem!")

    # Cache this scenario's report so it can be compared/evaluated against other scenarios later.
    st.session_state.evaluated_reports[scenario_choice.name] = {
        "report": report,
        "duration_sec": metadata.total_duration_sec,
        # Full transcript kept so the judge can build a multi-region (start/middle/end) excerpt.
        # The engine bounds how much it actually sends via Config.JUDGE_EXCERPT_CHARS.
        "raw_excerpt": raw_text,
        "input_fingerprint": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    # Remember which scenario to show by default; the report is rendered OUTSIDE this block
    # (below) so it survives Streamlit reruns triggered by any later widget interaction.
    st.session_state.active_report_scenario = scenario_choice.name

# ---------------------------------------------------------------------------
# PERSISTENT REPORT VIEW — rendered from the cache so it does NOT disappear when
# the user clicks anything else (Streamlit reruns the whole script each time).
# ---------------------------------------------------------------------------
if st.session_state.evaluated_reports:
    st.divider()
    st.header("📄 Raport z analizy")
    scenario_names = list(st.session_state.evaluated_reports.keys())
    default_idx = scenario_names.index(st.session_state.active_report_scenario) \
        if st.session_state.get("active_report_scenario") in scenario_names else 0
    chosen = st.selectbox(
        "Pokaż raport dla scenariusza:",
        options=scenario_names,
        index=default_idx,
        key="report_view_selector"
    )
    render_report(st.session_state.evaluated_reports[chosen]["report"])

# =========================================================================
# SEKCJA WSADOWA: uruchom wiele scenariuszy na wybranych modelach, oceń i pobierz raport
# =========================================================================
st.divider()
st.header("🧬 Uruchomienie wsadowe (scenariusze 1–4/5)")

# Recovery: reload results saved to disk (e.g. after a crash mid-batch) into the session.
_lc, _rc = st.columns([1, 2])
with _lc:
    if st.button(f"📂 Wczytaj zapisane wyniki z dysku ({Config.RUNS_DIR})"):
        n = _load_runs_from_disk()
        st.success(f"Wczytano {n} nowych wyników z dysku.") if n else st.info("Brak nowych wyników na dysku.")
with _rc:
    st.caption(f"Wyniki w pamięci sesji: **{len(st.session_state.runs)}** "
               f"(zapis na dysku: `{Config.RUNS_DIR}/run_*.json`).")

if not st.session_state.zip_data["is_valid"]:
    st.info("Najpierw wczytaj paczkę ZIP powyżej, aby uruchomić tryb wsadowy.")
else:
    _has_presentation = bool(st.session_state.zip_data.get("timeline")) or bool(
        st.session_state.zip_data.get("slide_summaries"))
    _default_scenarios = scenarios_for_batch(_has_presentation)
    st.caption(
        f"Modele z panelu bocznego: Hegemon=`{hegemon_model}`, Merytoryczny=`{factual_model}`, "
        f"Językowy=`{linguistic_model}`. Prezentacja wykryta: {'TAK' if _has_presentation else 'nie'}."
    )
    chosen_scenarios = st.multiselect(
        "Scenariusze do uruchomienia (kolejno):",
        options=[e for e in ExperimentScenario],
        default=_default_scenarios,
        format_func=lambda x: f"{x.value} - {x.name}"
    )
    auto_judge = st.checkbox("Po uruchomieniu od razu oceń i porównaj (sędzia)", value=True)
    batch_judge_model = st.selectbox("Model sędziego (dla oceny wsadu):", options=Config.get_all_models(), index=0,
                                     key="batch_judge_model")
    batch_judge_cfg = _judge_config_controls("batch")
    batch_kb_pdf = st.file_uploader("(Opcjonalnie) PDF bazy wiedzy dla scenariuszy RAG (4/5)", type="pdf",
                                    key="batch_kb")

    if st.button("🚀 Uruchom wybrane scenariusze", disabled=not chosen_scenarios):
        # Preflight models once for the whole batch.
        missing = check_ollama_models([hegemon_model, factual_model, linguistic_model, Config.UTILITY_MODEL])
        if missing:
            st.error("❌ Brakuje modeli w Ollama: " + ", ".join(f"`{m}`" for m in missing)
                     + ". Pobierz je lub zmień wybór w panelu bocznym.")
            st.stop()

        _md = st.session_state.zip_data["metadata"]
        _fingerprint = hashlib.sha256(st.session_state.zip_data["raw_text"].encode("utf-8")).hexdigest()
        _kb_bytes = batch_kb_pdf.getvalue() if batch_kb_pdf is not None else None

        with st.status("Tryb wsadowy pracuje…", expanded=True) as bstatus:
            def _bprogress(m: str):
                bstatus.write(m)


            def _persist(r):
                # Incremental save: append to session state AND write to disk immediately, so a
                # later scenario crashing never loses already-completed results.
                st.session_state.runs.append(r)
                try:
                    import os
                    os.makedirs(Config.RUNS_DIR, exist_ok=True)
                    with open(os.path.join(Config.RUNS_DIR, f"run_{r.run_id}.json"), "w",
                              encoding="utf-8") as fh:
                        fh.write(r.model_dump_json(indent=2))
                except Exception as _e:
                    bstatus.write(f"   ⚠️ Nie udało się zapisać na dysk: {_e}")


            new_runs = run_batch(
                scenarios=chosen_scenarios,
                zip_data=st.session_state.zip_data,
                speaker_role=speaker_role, target_audience=target_audience,
                main_topic=main_topic, knowledge_level=knowledge_level,
                hegemon_model=hegemon_model, factual_model=factual_model, linguistic_model=linguistic_model,
                use_llmlingua=use_llmlingua_switch,
                input_fingerprint=_fingerprint,
                source_label=uploaded_zip.name if uploaded_zip is not None else "zip",
                knowledge_base_bytes=_kb_bytes,
                progress_cb=_bprogress,
                on_result=_persist,
            )

            batch_eval = None
            if auto_judge and len(new_runs) >= 1:
                bstatus.write("🧑‍⚖️ Sędzia ocenia wyniki wsadu…")
                eval_gateway = LLMGateway(ObservabilityManager())
                engine = EvaluationEngine(eval_gateway, batch_judge_model, **batch_judge_cfg)
                reports = {r.display_label(): r.report for r in new_runs}
                duration = max((r.duration_sec for r in new_runs), default=0.0)
                excerpt = st.session_state.zip_data.get("raw_text", "")
                batch_eval = asyncio.run(engine.evaluate(excerpt, reports, duration_sec=duration))
                eval_gateway.reset_session_telemetry()

            bstatus.update(label=f"Wsad zakończony: {len(new_runs)} scenariuszy.", state="complete")

        # Build downloadable artifact (JSON + Markdown summary).
        export = BatchExport(
            created_at=datetime.now(timezone.utc).isoformat(),
            source_label=uploaded_zip.name if uploaded_zip is not None else "zip",
            runs=new_runs,
            evaluation=batch_eval,
        )
        st.success(f"✅ Uruchomiono {len(new_runs)} scenariuszy. Pobierz raport poniżej.")
        st.download_button(
            "⬇️ Pobierz raport wsadu (JSON)",
            data=export.model_dump_json(indent=2),
            file_name=f"batch_{export.created_at[:19].replace(':', '-')}.json",
            mime="application/json"
        )
        st.download_button(
            "⬇️ Pobierz podsumowanie (Markdown)",
            data=_batch_markdown(export),
            file_name=f"batch_{export.created_at[:19].replace(':', '-')}.md",
            mime="text/markdown"
        )

# =========================================================================
# SEKCJA EWALUACJI / PORÓWNANIA SCENARIUSZY (LLM-as-judge + telemetria)
# =========================================================================
st.divider()
st.header("🧪 Ewaluacja i porównanie scenariuszy")

cached = st.session_state.evaluated_reports
if not cached:
    st.info("Uruchom co najmniej jeden scenariusz, aby zgromadzić raporty do porównania.")
else:
    st.caption(f"Zbuforowane raporty: {', '.join(cached.keys())}")
    all_models = Config.get_all_models()
    judge_model = st.selectbox("Model sędziego (Judge):", options=all_models, index=0)
    manual_judge_cfg = _judge_config_controls("manual")
    selected = st.multiselect(
        "Wybierz scenariusze do porównania (2+ dla preferencji parami):",
        options=list(cached.keys()),
        default=list(cached.keys())
    )
    if st.button("🔍 Oceń i porównaj") and selected:
        fingerprints = {cached[name].get("input_fingerprint") for name in selected}
        if len(fingerprints) > 1:
            st.error(
                "⚠️ Wybrane scenariusze pochodzą z RÓŻNYCH danych wejściowych (różne ZIP-y). "
                "Porównanie ma sens tylko dla tego samego wystąpienia. Uruchom scenariusze na tej samej paczce."
            )
            st.stop()
        reports = {name: cached[name]["report"] for name in selected}
        duration = max((cached[name]["duration_sec"] for name in selected), default=0.0)
        excerpt = next((cached[name]["raw_excerpt"] for name in selected), "")

        with st.spinner("Sędzia ocenia raporty..."):
            eval_obs = ObservabilityManager()
            eval_gateway = LLMGateway(eval_obs)
            engine = EvaluationEngine(eval_gateway, judge_model, **manual_judge_cfg)
            eval_report = asyncio.run(engine.evaluate(excerpt, reports, duration_sec=duration))

        st.subheader("📊 Jakość vs. koszt")
        rows = []
        for se in eval_report.per_scenario:
            rows.append({
                "Scenariusz": se.scenario_name,
                "Jakość (0-50)": se.rubric_total,
                "Konkretność rad": se.rubric.actionability,
                "Szczegółowość": se.rubric.specificity,
                "Trafność merytoryczna": se.rubric.correctness,
                "Ton": se.rubric.tone,
                "Osadzenie w faktach": se.rubric.groundedness,
                "Tokeny (we+wy)": se.total_tokens_in + se.total_tokens_out,
                "Koszt ($)": round(se.total_cost_usd, 5),
                # Cost per quality point (cost / rubric_total). LOWER = more quality per dollar.
                # This operationalizes the money-vs-quality tradeoff: a scenario that spends more
                # but scores proportionally higher isn't necessarily worse value.
                "Koszt / pkt jakości ($)": round(se.total_cost_usd / se.rubric_total, 6) if se.rubric_total else None,
                "Gęstość WE (znaki/token)": se.input_token_density,
                "Gęstość WY (znaki/token)": se.output_token_density,
                "Pokrycie (regiony)": ", ".join(
                    f"{x:.2f}" for x in se.positional_recall) if se.positional_recall else "brak znaczników",
                "Wierność reduce": ", ".join(f"{x:.2f}" for x in se.reduce_fidelity) if se.reduce_fidelity else "—",
                "Zagubienie w środku": "⚠️ TAK" if se.lost_in_middle_flag else "nie",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # --- Option 1: Δ vs. baseline — direct A/B comparison against a chosen reference scenario ---
        per = eval_report.per_scenario
        if len(per) >= 2:
            st.subheader("📐 Porównanie do bazy (Δ)")
            st.caption(
                "Wybierz scenariusz bazowy (np. Monolith Naked). Pozostałe pokazane jako RÓŻNICA względem niego "
                "— dodatnia jakość/osadzenie = lepiej, dodatni koszt = drożej."
            )
            names = [se.scenario_name for se in per]
            base_name = st.selectbox("Scenariusz bazowy:", options=names, index=0, key="ab_baseline")
            base = next(se for se in per if se.scenario_name == base_name)
            delta_rows = []
            for se in per:
                if se.scenario_name == base_name:
                    continue
                delta_rows.append({
                    "Scenariusz": se.scenario_name,
                    "Δ Jakość (0-50)": round(se.rubric_total - base.rubric_total, 1),
                    "Δ Osadzenie w faktach": se.rubric.groundedness - base.rubric.groundedness,
                    "Δ Trafność": se.rubric.correctness - base.rubric.correctness,
                    "Δ Koszt ($)": round(se.total_cost_usd - base.total_cost_usd, 5),
                    "Lost-in-middle (baza→ten)": f"{'TAK' if base.lost_in_middle_flag else 'nie'} → "
                                                 f"{'TAK' if se.lost_in_middle_flag else 'nie'}",
                })
            if delta_rows:
                st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)

        # --- Option 2: what the judge actually saw + its reasoning (auditability) ---
        st.subheader("🔎 Dowody sędziego (audyt osadzenia)")
        st.caption(
            "Sprawdź, DLACZEGO sędzia dał daną ocenę osadzenia — jego uzasadnienie oraz materiał, który widział.")
        for se in per:
            with st.expander(f"{se.scenario_name} — groundedness {se.rubric.groundedness}/10, "
                             f"correctness {se.rubric.correctness}/10"):
                if se.rubric.justification:
                    st.markdown(f"**Uzasadnienie sędziego:** {se.rubric.justification}")
                if se.judge_evidence:
                    st.text_area("Materiał przekazany sędziemu", se.judge_evidence, height=300,
                                 key=f"evidence_{se.scenario_name}")
                else:
                    st.info("Brak zapisanego materiału dowodowego.")

        if eval_report.pairwise:
            st.subheader("⚔️ Preferencje parami")
            for pref in eval_report.pairwise:
                st.markdown(f"- **Zwycięzca: {pref.winner}** — {pref.reason}")

        st.caption(
            f"Tokeny sędziego: {eval_report.judge_tokens_in + eval_report.judge_tokens_out}. {eval_report.summary}"
        )
        eval_gateway.reset_session_telemetry()
