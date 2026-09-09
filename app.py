import asyncio
import hashlib
import json
import pandas as pd
import streamlit as st
import zipfile
from datetime import datetime, timezone

from core.batch_runner import run_batch, scenarios_for_batch
from core.config_loader import Config
from core.llm_gateway import LLMGateway
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

if 'evaluated_reports' not in st.session_state:
    st.session_state.evaluated_reports = {}

if 'active_report_scenario' not in st.session_state:
    st.session_state.active_report_scenario = None

if 'runs' not in st.session_state:
    st.session_state.runs = []

if 'map_results' not in st.session_state:
    st.session_state.map_results = []


def read_golden_set(file_obj) -> str:
    if file_obj is None:
        return ""
    name = file_obj.name.lower()
    if name.endswith(".json"):
        try:
            return json.dumps(json.loads(file_obj.getvalue().decode('utf-8')), indent=2, ensure_ascii=False)
        except:
            return file_obj.getvalue().decode('utf-8', errors='ignore')
    elif name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_obj.getvalue(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            st.warning(f"Błąd odczytu PDF (upewnij się że masz zainstalowane pymupdf): {e}")
            return ""
    else:
        return file_obj.getvalue().decode('utf-8', errors='ignore')


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
                        chunks_data.append(json.loads(z.read(f).decode('utf-8')))
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
            st.success(f"✅ Wczytano paczkę: {len(chunks_data)} chunków.")
            return True
    except Exception as e:
        st.error(f"❌ Błąd przetwarzania paczki ZIP: {e}")
        return False


def _load_maps_from_disk() -> int:
    import os
    from models.schemas import MapResult
    existing_ids = {m.map_id for m in st.session_state.map_results}
    added = 0
    directory = Config.MAPS_DIR
    if not os.path.isdir(directory): return 0
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"): continue
        try:
            with open(os.path.join(directory, fname), encoding="utf-8") as fh:
                m = MapResult.model_validate_json(fh.read())
            if m.map_id not in existing_ids:
                st.session_state.map_results.append(m)
                existing_ids.add(m.map_id)
                added += 1
        except:
            continue
    return added


def _load_runs_from_disk() -> int:
    import os
    from models.schemas import RunResult
    existing_ids = {r.run_id for r in st.session_state.runs}
    added = 0
    directory = Config.RUNS_DIR
    if not os.path.isdir(directory): return 0
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"): continue
        try:
            with open(os.path.join(directory, fname), encoding="utf-8") as fh:
                r = RunResult.model_validate_json(fh.read())
            if r.run_id not in existing_ids:
                st.session_state.runs.append(r)
                existing_ids.add(r.run_id)
                added += 1
        except:
            continue
    return added


def _judge_config_controls(key_prefix: str) -> dict:
    with st.expander("⚙️ Konfiguracja sędziego (osadzenie / groundedness)"):
        excerpt_chars = st.slider("Rozmiar fragmentu transkrypcji (znaki)", 0, 30000, Config.JUDGE_EXCERPT_CHARS,
                                  step=1000, key=f"{key_prefix}_excerpt_chars")
        excerpt_regions = st.slider("Liczba regionów", 1, 5, Config.JUDGE_EXCERPT_REGIONS,
                                    key=f"{key_prefix}_excerpt_regions")
        probe_timestamps = st.slider("Sondy czasowe [MM:SS]", 0, 20, Config.JUDGE_PROBE_TIMESTAMPS,
                                     key=f"{key_prefix}_probe_ts")
        probe_window = st.slider("Okno sondy (znaki)", 100, 2000, Config.JUDGE_PROBE_WINDOW_CHARS, step=100,
                                 key=f"{key_prefix}_probe_win")
        focus = st.text_area("Szczególny nacisk dla sędziego", value="", key=f"{key_prefix}_focus")
    return {
        "excerpt_chars": excerpt_chars, "excerpt_regions": excerpt_regions,
        "probe_timestamps": probe_timestamps, "probe_window_chars": probe_window,
        "focus_instruction": focus,
    }


def _batch_markdown(export) -> str:
    lines = [f"# Raport wsadowy — {export.source_label}", f"Utworzono: {export.created_at}", "", "## Uruchomienia"]
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
    return "\n".join(lines)


def _fmt_score(value) -> str:
    return f"{value}/100" if value is not None else "nie dotyczy"


def _reports_from_uploaded_json(raw: str, filename: str) -> list:
    from models.schemas import FinalReport, RunResult, BatchExport
    out = []
    try:
        be = BatchExport.model_validate_json(raw)
        if be.runs:
            for rr in be.runs: out.append((f"{rr.scenario_name} · {rr.models.hegemon_model.split('/')[-1]}", rr.report))
            return out
    except:
        pass
    try:
        rr = RunResult.model_validate_json(raw)
        return [(f"{rr.scenario_name} · {rr.models.hegemon_model.split('/')[-1]}", rr.report)]
    except:
        pass
    try:
        fr = FinalReport.model_validate_json(raw)
        return [(filename.rsplit(".", 1)[0], fr)]
    except:
        return []


def _report_markdown(report, title: str = "Raport") -> str:
    a, fb, sc, tel = report.analysis, report.feedback, report.scorecard, report.telemetry
    lines = [f"# {title}", ""]
    if sc is not None:
        parts = [f"**Ocena łączna:** {sc.overall_score}/100 — {sc.readiness_verdict}"]
        if sc.factual_score is not None: parts.append(f"Merytoryka: {sc.factual_score}/100")
        if sc.linguistic_score is not None: parts.append(f"Język: {sc.linguistic_score}/100")
        if sc.slide_coverage_score is not None: parts.append(f"Pokrycie slajdów: {sc.slide_coverage_score}/100")
        lines += ["  |  ".join(parts), ""]

    lines += ["## Podsumowanie merytoryczne", a.factual_summary or "_(brak)_", ""]
    lines += ["## Analiza językowa", a.linguistic_summary or "_(brak)_", ""]
    if a.missed_context: lines += ["## Pominięte wątki", *[f"- {c}" for c in a.missed_context], ""]
    if a.unverified_claims: lines += ["## ⚠️ Twierdzenia wymagające weryfikacji",
                                      *[f"- {c}" for c in a.unverified_claims], ""]

    lines += ["## Feedback mentorski", fb.executive_summary_markdown or "_(brak)_", ""]
    if fb.strengths: lines += ["### Mocne strony", *[f"- {s}" for s in fb.strengths], ""]
    if fb.areas_for_improvement: lines += ["### Obszary do poprawy", *[f"- {x}" for x in fb.areas_for_improvement], ""]
    if fb.actionable_tips: lines += ["### Wskazówki", *[f"- {t}" for t in fb.actionable_tips], ""]
    if fb.overall_message: lines += ["### Główne przesłanie", fb.overall_message, ""]

    lines += ["---",
              f"_Telemetria: koszt ${tel.total_cost_usd:.4f}, tokeny {tel.total_tokens_in + tel.total_tokens_out}._"]
    return "\n".join(lines)


def render_report(report, key: str = "report", title: str = "Raport z analizy"):
    _dc1, _dc2 = st.columns(2)
    with _dc1:
        st.download_button("⬇️ Pobierz ten raport (Markdown)", data=_report_markdown(report, title),
                           file_name=f"{key}.md", mime="text/markdown")
    with _dc2:
        st.download_button("⬇️ Pobierz do sędziego (JSON)", data=report.model_dump_json(indent=2),
                           file_name=f"{key}.report.json", mime="application/json")

    sc = report.scorecard
    if sc:
        st.subheader(f"🏁 Ocena łączna: {sc.overall_score}/100 — {sc.readiness_verdict}")
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Merytoryka", _fmt_score(sc.factual_score))
        sm2.metric("Język", _fmt_score(sc.linguistic_score))
        if sc.slide_coverage_score is not None: sm3.metric("Pokrycie slajdów", _fmt_score(sc.slide_coverage_score))

    tab1, tab2, tab3 = st.tabs(["🧠 Analiza i Detale", "💡 Feedback", "📊 Raport Kosztowy z Roju"])

    with tab1:
        st.info(report.analysis.factual_summary)
        st.info(report.analysis.linguistic_summary)
        if report.analysis.unverified_claims:
            st.warning("**⚠️ Twierdzenia wymagające weryfikacji**\n" + "\n".join(
                f"- {c}" for c in report.analysis.unverified_claims))

    with tab2:
        if report.feedback.executive_summary_markdown: st.markdown(report.feedback.executive_summary_markdown)
        st.divider()
        c1, c2 = st.columns(2)
        for s in report.feedback.strengths: c1.markdown(f"✅ {s}")
        for a in report.feedback.areas_for_improvement: c2.markdown(f"🔧 {a}")

    with tab3:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sumaryczny Koszt", f"${report.telemetry.total_cost_usd:.4f}")
        m2.metric("Suma Tokenów", f"{report.telemetry.total_tokens_in + report.telemetry.total_tokens_out}")
        m3.metric("Fazy Map / Reduce", f"{report.telemetry.map_phases_count} / {report.telemetry.reduce_phases_count}")
        if report.telemetry.phase_details:
            df = pd.DataFrame([d.model_dump() for d in report.telemetry.phase_details])
            st.dataframe(df, hide_index=True, use_container_width=True)


st.sidebar.header("⚙️ Konfiguracja Systemu")
all_models = Config.get_all_models()

scenario_choice = st.sidebar.selectbox("Wybierz Scenariusz Badawczy:", options=[e for e in ExperimentScenario],
                                       format_func=lambda x: f"{x.value} - {x.name}")
use_llmlingua_switch = st.sidebar.checkbox("Użyj LLMLingua (Kompresja Entropijna)", value=False)

st.sidebar.subheader("Modele dla ról")
_HEGEMON_DEFAULT = "ollama/llama3.1:70b"
_FACTUAL_DEFAULT = "ollama/Speakleash/bielik-11b-v3.0-instruct:Q5_K_M"
_LINGUISTIC_DEFAULT = "ollama/llama3.1:8b"

hegemon_model = st.sidebar.selectbox("Hegemon (Reduce / Monolith):", options=all_models,
                                     index=all_models.index(_HEGEMON_DEFAULT) if _HEGEMON_DEFAULT in all_models else 0)
factual_model = st.sidebar.selectbox("Agent Merytoryczny (Map):", options=all_models,
                                     index=all_models.index(_FACTUAL_DEFAULT) if _FACTUAL_DEFAULT in all_models else 0)
linguistic_model = st.sidebar.selectbox("Agent Językowy (Map):", options=all_models, index=all_models.index(
    _LINGUISTIC_DEFAULT) if _LINGUISTIC_DEFAULT in all_models else 0)

st.subheader("📂 Krok 1: Wczytaj dane wejściowe")
uploaded_zip = st.file_uploader("1. Załaduj wygenerowaną paczkę ZIP z danymi", type="zip")
if uploaded_zip and st.button("Sprawdź i załaduj strukturę paczki"):
    process_uploaded_zip(uploaded_zip)

uploaded_kb_pdf = st.file_uploader("2. (Opcjonalnie) Baza wiedzy PDF dla RAG", type="pdf")

st.divider()
st.subheader("📋 Krok 2: Weryfikacja Metadanych Wykładu")
md = st.session_state.zip_data["metadata"]

with st.form("metadata_form"):
    col1, col2 = st.columns(2)
    with col1:
        speaker_role = st.text_input("Rola prelegenta", value=md.get("speaker_role", ""))
        target_audience = st.text_input("Grupa docelowa", value=md.get("target_audience", ""))
        kl_options = ["Brak", "Podstawowy", "Średni", "Zaawansowany", "Ekspert"]
        knowledge_level = st.selectbox("Poziom wiedzy odbiorców", kl_options,
                                       index=kl_options.index(md.get("knowledge_level", "Podstawowy")) if md.get(
                                           "knowledge_level", "Podstawowy") in kl_options else 1)
    with col2:
        main_topic = st.text_input("Główny temat", value=md.get("main_topic", ""))
        st.markdown(f"**Czas:** {md.get('total_duration_sec', 0)}s | **Słowa:** {md.get('total_words', 0)}")

    submitted = st.form_submit_button("🚀 Uruchom Ewaluację Przemówienia",
                                      disabled=not st.session_state.zip_data["is_valid"])

if submitted and st.session_state.zip_data["is_valid"]:
    metadata = LectureMetadata(
        speaker_role=speaker_role, target_audience=target_audience, main_topic=main_topic,
        knowledge_level=knowledge_level, total_duration_sec=md.get("total_duration_sec", 0.0),
        total_words=md.get("total_words", 0)
    )
    system_config = SystemConfiguration(
        scenario=scenario_choice, hegemon_model=hegemon_model,
        agent_models=AgentModelsConfig(factual_model=factual_model, linguistic_model=linguistic_model),
        use_tools=scenario_choice in [ExperimentScenario.SWARM_NAIVE_RAG_WEB,
                                      ExperimentScenario.SWARM_PRESENTATION_RAG_WEB],
        use_llmlingua=use_llmlingua_switch
    )

    with st.status(f"Orkiestrator pracuje ({scenario_choice.name})...", expanded=True) as status:
        orchestrator = Orchestrator(system_config, gateway=LLMGateway(ObservabilityManager()), progress_cb=status.write)
        try:
            report = orchestrator.execute_pipeline(
                metadata=metadata, raw_text=st.session_state.zip_data["raw_text"],
                formatted_text=st.session_state.zip_data["formatted_text"],
                chunks=[ChunkPayload(**c) for c in st.session_state.zip_data["chunks"]],
                timeline=TimelinePayload(**st.session_state.zip_data["timeline"]) if st.session_state.zip_data.get(
                    "timeline") else None,
                slide_summaries={k: SlideSummary(**v) for k, v in
                                 st.session_state.zip_data.get("slide_summaries", {}).items()},
                knowledge_base_bytes=uploaded_kb_pdf.getvalue() if uploaded_kb_pdf else None
            )
            st.session_state.evaluated_reports[scenario_choice.name] = {
                "report": report, "duration_sec": metadata.total_duration_sec,
                "total_words": metadata.total_words, "raw_excerpt": st.session_state.zip_data["raw_text"],
                "input_fingerprint": hashlib.sha256(st.session_state.zip_data["raw_text"].encode("utf-8")).hexdigest(),
            }
            status.update(label="Zakończono sukcesem!", state="complete")
        except Exception as e:
            status.update(label="Błąd", state="error");
            st.error(f"❌ {e}");
            st.stop()

if st.session_state.evaluated_reports:
    chosen = st.selectbox("Pokaż raport dla scenariusza:", options=list(st.session_state.evaluated_reports.keys()))
    render_report(st.session_state.evaluated_reports[chosen]["report"], key=f"view_{chosen}",
                  title=f"Raport — {chosen}")

# =========================================================================
# SEKCJA WSADOWA
# =========================================================================
st.divider()
st.header("🧬 Uruchomienie wsadowe (scenariusze 1–4/5)")

_lc, _rc = st.columns([1, 2])
with _lc:
    if st.button(f"📂 Wczytaj zapisane wyniki z dysku ({Config.RUNS_DIR})"):
        st.success(f"Wczytano {_load_runs_from_disk()} nowych wyników z dysku.")
with _rc: st.caption(f"Wyniki w pamięci sesji: **{len(st.session_state.runs)}**")

if st.session_state.zip_data["is_valid"]:
    chosen_scenarios = st.multiselect(
        "Scenariusze do uruchomienia:", options=[e for e in ExperimentScenario],
        default=scenarios_for_batch(bool(st.session_state.zip_data.get("timeline"))),
        format_func=lambda x: f"{x.value} - {x.name}"
    )

    if st.button("🚀 Uruchom wybrane scenariusze wsadowo", disabled=not chosen_scenarios):
        with st.status("Tryb wsadowy pracuje…", expanded=True) as bstatus:
            def _persist(r):
                st.session_state.runs.append(r)
                try:
                    import os
                    os.makedirs(Config.RUNS_DIR, exist_ok=True)
                    with open(os.path.join(Config.RUNS_DIR, f"run_{r.run_id}.json"), "w", encoding="utf-8") as fh:
                        fh.write(r.model_dump_json(indent=2))
                except:
                    pass


            new_runs = run_batch(
                scenarios=chosen_scenarios, zip_data=st.session_state.zip_data, speaker_role=speaker_role,
                target_audience=target_audience, main_topic=main_topic, knowledge_level=knowledge_level,
                hegemon_model=hegemon_model, factual_model=factual_model, linguistic_model=linguistic_model,
                use_llmlingua=use_llmlingua_switch,
                input_fingerprint=hashlib.sha256(st.session_state.zip_data["raw_text"].encode("utf-8")).hexdigest(),
                source_label=uploaded_zip.name if uploaded_zip else "zip",
                progress_cb=bstatus.write, on_result=_persist,
            )
            bstatus.update(label=f"Wsad zakończony: {len(new_runs)} scenariuszy.", state="complete")

        export = BatchExport(created_at=datetime.now(timezone.utc).isoformat(),
                             source_label=uploaded_zip.name if uploaded_zip else "zip", runs=new_runs)
        st.success(f"✅ Uruchomiono {len(new_runs)} scenariuszy.")
        st.download_button("⬇️ Pobierz raport wsadu (JSON)", data=export.model_dump_json(indent=2),
                           file_name="batch.json")

# =========================================================================
# SEKCJA MAP/REDUCE: Uruchom MAP raz i testuj różnych Hegemonów
# =========================================================================
st.divider()
st.header("🧩 Faza MAP osobno + porównanie reduktorów")

_mlc, _mrc = st.columns([1, 2])
with _mlc:
    if st.button(f"📂 Wczytaj zapisane fazy MAP ({Config.MAPS_DIR})"):
        st.success(f"Wczytano {_load_maps_from_disk()} nowych faz MAP.")
with _mrc: st.caption(f"Fazy MAP w pamięci: {len(st.session_state.map_results)}")

if st.session_state.zip_data["is_valid"]:
    map_scenario = st.selectbox("Scenariusz dla MAP:",
                                options=[ExperimentScenario.SWARM_NAIVE_NO_RAG, ExperimentScenario.SWARM_NAIVE_RAG_WEB,
                                         ExperimentScenario.SWARM_PRESENTATION_RAG_WEB],
                                format_func=lambda x: f"{x.value} - {x.name}")
    if st.button("🧠 Uruchom TYLKO fazę MAP"):
        with st.status("Faza MAP pracuje…", expanded=True) as mstatus:
            _orch = Orchestrator(SystemConfiguration(scenario=map_scenario, hegemon_model=hegemon_model,
                                                     agent_models=AgentModelsConfig(factual_model=factual_model,
                                                                                    linguistic_model=linguistic_model)),
                                 gateway=LLMGateway(ObservabilityManager()), progress_cb=mstatus.write)
            try:
                _mr = _orch.execute_map_only(
                    metadata=LectureMetadata(
                        total_duration_sec=st.session_state.zip_data["metadata"].get("total_duration_sec", 0.0),
                        total_words=st.session_state.zip_data["metadata"].get("total_words", 0)),
                    chunks=[ChunkPayload(**c) for c in st.session_state.zip_data.get("chunks", [])],
                    timeline=TimelinePayload(**st.session_state.zip_data["timeline"]) if st.session_state.zip_data.get(
                        "timeline") else None,
                    slide_summaries={k: SlideSummary(**v) for k, v in
                                     st.session_state.zip_data.get("slide_summaries", {}).items()},
                    input_fingerprint=hashlib.sha256(st.session_state.zip_data["raw_text"].encode("utf-8")).hexdigest(),
                )
                st.session_state.map_results.append(_mr)
                mstatus.update(label="Faza MAP zakończona.", state="complete")
            except Exception as e:
                mstatus.update(label="Błąd MAP", state="error");
                st.stop()

if st.session_state.map_results:
    _chosen_map_label = st.selectbox("Wybierz zapisaną fazę MAP:",
                                     options=list({m.display_label(): m for m in st.session_state.map_results}.keys()))
    _reduce_hegemon = st.selectbox("Model Hegemona:", options=Config.get_all_models())
    if st.button("🏛️ Uruchom REDUCE"):
        _mr = {m.display_label(): m for m in st.session_state.map_results}[_chosen_map_label]
        with st.status(f"REDUCE ({_reduce_hegemon})…", expanded=True) as rstatus:
            _orch = Orchestrator(
                SystemConfiguration(scenario=ExperimentScenario[_mr.scenario_name], hegemon_model=_reduce_hegemon,
                                    agent_models=AgentModelsConfig(factual_model=_mr.factual_model,
                                                                   linguistic_model=_mr.linguistic_model)),
                gateway=LLMGateway(ObservabilityManager()), progress_cb=rstatus.write)
            _report = _orch.execute_reduce_from_map(_mr)
            rstatus.update(label="REDUCE zakończony.", state="complete")
            render_report(_report, key="reduce_latest", title=_mr.scenario_name)

# =========================================================================
# SEKCJA EWALUACJI / PORÓWNANIA SCENARIUSZY (MLOps Leaderboard)
# =========================================================================
st.divider()
st.header("🏆 MLOps Leaderboard: Globalny Ranking Modeli i Architektur")

cached = st.session_state.evaluated_reports
if not cached:
    st.info("Uruchom co najmniej jeden scenariusz, aby zgromadzić raporty do porównania.")
else:
    judge_model = st.selectbox("Model sędziego (Judge):", options=Config.get_all_models(), index=0)
    manual_judge_cfg = _judge_config_controls("manual")

    st.subheader("🎯 Złoty Wzorzec (Ground Truth)")
    col_gt1, col_gt2 = st.columns(2)
    with col_gt1:
        exp_factual = st.slider("Oczekiwana Merytoryka", 0.0, 100.0, 70.0, 0.5)
    with col_gt2:
        exp_linguistic = st.slider("Oczekiwany Język", 0.0, 100.0, 30.0, 0.5)

    st.subheader("📁 Pliki referencyjne błędów (JSON Golden Sets)")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        golden_factual_file = st.file_uploader("Złoty Wzorzec MERYTORYCZNY (JSON)", type=["json"],
                                               key="leaderboard_factual")
    with col_f2:
        golden_linguistic_file = st.file_uploader("Złoty Wzorzec LINGWISTYCZNY (JSON)", type=["json"],
                                                  key="leaderboard_linguistic")

    selected = st.multiselect("Wybierz scenariusze do porównania:", options=list(cached.keys()),
                              default=list(cached.keys()))

    if st.button("🚀 Wygeneruj Ranking (LLM-as-a-Judge)") and selected:
        txt_factual = read_golden_set(golden_factual_file)
        txt_linguistic = read_golden_set(golden_linguistic_file)

        reports = {name: cached[name]["report"] for name in selected}
        duration = max((cached[name]["duration_sec"] for name in selected), default=0.0)
        total_words = max((cached[name].get("total_words", 0) for name in selected), default=0)
        excerpt = next((cached[name]["raw_excerpt"] for name in selected), "")

        with st.spinner("Sędzia ocenia raporty..."):
            eval_gateway = LLMGateway(ObservabilityManager())
            engine = EvaluationEngine(eval_gateway, judge_model, **manual_judge_cfg)
            eval_report, extra_metrics = asyncio.run(engine.evaluate(
                transcript_excerpt=excerpt, reports=reports, duration_sec=duration,
                total_words=total_words, expected_factual=exp_factual, expected_linguistic=exp_linguistic,
                golden_factual=txt_factual, golden_linguistic=txt_linguistic
            ))

        # Agregacja zwycięstw H2H (tylko na potrzeby sortowania pod maską)
        h2h_wins = {name: 0 for name in selected}
        if eval_report.pairwise:
            for pref in eval_report.pairwise:
                if pref.winner in h2h_wins:
                    h2h_wins[pref.winner] += 1

        st.subheader("🥇 Tabela Wyników")
        rows = []
        for se in eval_report.per_scenario:
            ext = extra_metrics.get(se.scenario_name, {})
            costs = ext.get("costs", {})

            recall_pct = ext.get("error_recall_pct", -1.0)
            recall_str = "Brak danych" if recall_pct < 0 else f"{recall_pct}%"
            crit_pct = ext.get("error_recall_critical_pct", -1.0)
            crit_str = "—" if crit_pct is None or crit_pct < 0 else f"{crit_pct}%"

            actual_model = ext.get("actual_hegemon_model", "") or costs.get("actual_hegemon_model", "")
            actual_short = actual_model.split("/")[-1] if actual_model else "—"
            fb_used = ext.get("fallback_used", False)
            model_cell = f"⚠️ {actual_short} (fallback)" if fb_used else actual_short

            missing = ext.get("missing_sections", [])
            miss_cell = "✓ komplet" if not missing else f"⚠️ brak: {', '.join(missing)}"
            lim_cell = "⚠️ TAK" if ext.get("lost_in_middle") else "—"

            rows.append({
                "Architektura / Model": se.scenario_name,
                "🤖 Model (faktyczny)": model_cell,
                "🧩 Typ": ext.get("scenario_kind", ""),
                "🏆 Jakość (0-50)": se.rubric_total,
                "🧾 Kompletność": miss_cell,
                "🕳️ Lost-in-middle": lim_cell,
                "🎯 Odchylenie (RMSE)": ext.get("rmse", 0.0),
                "🎯 Wykryte błędy (%)": recall_str,
                "🎯 CRIT/HIGH (%)": crit_str,
                "🔤 TPW (Narzut)": ext.get("tpw", 0.0),
                "📦 Gęst. Meryt. (zn/tok)": ext.get("factual_density", 0.0),
                "📦 Gęst. Ling. (zn/tok)": ext.get("linguistic_density", 0.0),
                "📦 Tokeny MAP": costs.get("prior_tokens_total", 0),
                "📦 Hegemon IN (max)": costs.get("hegemon_tokens_in", 0),
                "📦 Hegemon OUT": costs.get("hegemon_tokens_out", 0),
                "📉 Koszt MAP ($)": costs.get("map_total_usd", 0.0),
                "📈 Koszt Hegemona ($)": costs.get("reduce_usd", 0.0),
                "Osadzenie": se.rubric.groundedness,
                "Wygrane H2H": h2h_wins.get(se.scenario_name, 0),  # Ukryta kolumna do sortowania
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=["Wygrane H2H", "🏆 Jakość (0-50)", "🎯 Odchylenie (RMSE)"],
                                ascending=[False, False, True])
            # Usuwamy H2H przed wyświetleniem, żeby nie śmieciło widoku
            df = df.drop(columns=["Wygrane H2H"])

            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "🏆 Jakość (0-50)": st.column_config.NumberColumn(format="%d/50"),
                "📉 Koszt MAP ($)": st.column_config.NumberColumn(format="$%.5f"),
                "📈 Koszt Hegemona ($)": st.column_config.NumberColumn(format="$%.5f")
            })

        st.subheader("🔎 Dowody sędziego (Audyt Ugruntowania)")
        for se in eval_report.per_scenario:
            with st.expander(f"{se.scenario_name} — Osadzenie {se.rubric.groundedness}/10"):
                st.markdown(f"**Uzasadnienie (Wykrywalność błędów):**\n\n{se.rubric.justification}")

# =========================================================================
# SEKCJA: WGRAJ ZAPISANE RAPORTY I OCEŃ SĘDZIĄ (cross-session compare)
# =========================================================================
st.divider()
st.header("📤 Wgraj zapisane raporty do porównania sędzią")

uploaded_reports = st.file_uploader("Pliki JSON raportów:", type="json", accept_multiple_files=True, key="judge_upload")
up_judge_model = st.selectbox("Model sędziego:", options=Config.get_all_models(), index=0, key="upload_judge_model")
up_judge_cfg = _judge_config_controls("upload")

st.subheader("📝 Pełna transkrypcja i metadane (dla groundedness, sond czasowych i lost-in-the-middle)")
st.caption("Wgraj CAŁĄ transkrypcję (full_raw_text.txt) — sędzia dostanie z niej wieloregionowy fragment "
           "i sondy czasowe. Metadane (metadata.json) dostarczą liczbę słów i czas trwania automatycznie, "
           "więc nie trzeba wpisywać ich ręcznie.")
col_tx1, col_tx2 = st.columns(2)
with col_tx1:
    up_transcript_file = st.file_uploader("Pełna transkrypcja (.txt)", type=["txt"], key="upload_transcript_file")
with col_tx2:
    up_metadata_file = st.file_uploader("Metadane nagrania (metadata.json)", type=["json"], key="upload_metadata_file")

# Metadane: automatyczne total_words i duration (zastępują ręczne pola).
_up_meta = {}
if up_metadata_file is not None:
    try:
        _up_meta = json.loads(up_metadata_file.getvalue().decode("utf-8"))
    except Exception as e:
        st.warning(f"Nie udało się odczytać metadata.json: {e}")

# Transkrypcja: preferuj wgrany plik, w ostateczności pole tekstowe poniżej.
up_transcript_text = ""
if up_transcript_file is not None:
    try:
        up_transcript_text = up_transcript_file.getvalue().decode("utf-8", errors="ignore")
        st.success(f"Wczytano transkrypcję: {len(up_transcript_text)} znaków.")
    except Exception as e:
        st.warning(f"Nie udało się odczytać transkrypcji: {e}")

col_up1, col_up2 = st.columns(2)
with col_up1: up_exp_factual = st.slider("Oczekiwana Merytoryka (Upload)", 0.0, 100.0, 70.0, 0.5)
with col_up2: up_exp_linguistic = st.slider("Oczekiwany Język (Upload)", 0.0, 100.0, 30.0, 0.5)

# total_words / duration: z metadanych, jeśli są; inaczej policz ze wgranej transkrypcji;
# ręczne pole pojawia się TYLKO jako fallback, gdy brak obu źródeł.
_meta_words = int(_up_meta.get("total_words", 0) or 0)
_meta_dur = float(_up_meta.get("total_duration_sec", 0.0) or 0.0)
if _meta_words:
    up_total_words = _meta_words
elif up_transcript_text:
    up_total_words = len(up_transcript_text.split())
else:
    up_total_words = st.number_input("Suma słów w nagraniu (fallback dla TPW — brak metadata.json)",
                                     min_value=0, value=0)
up_duration_sec = _meta_dur  # 0.0 gdy brak — sondy/positional recall wtedy nieaktywne
st.caption(f"Do obliczeń: słowa = **{up_total_words}**, czas = **{up_duration_sec:.0f}s** "
           f"({'z metadata.json' if _meta_words else ('policzone z transkrypcji' if up_transcript_text else 'brak — podaj ręcznie')}).")

st.subheader("📁 Pliki referencyjne błędów (JSON Golden Sets) dla Uploadu")
col_uf1, col_uf2 = st.columns(2)
with col_uf1: up_golden_factual_file = st.file_uploader("Złoty Wzorzec MERYTORYCZNY (JSON)", type=["json"],
                                                        key="upload_factual")
with col_uf2: up_golden_linguistic_file = st.file_uploader("Złoty Wzorzec LINGWISTYCZNY (JSON)", type=["json"],
                                                           key="upload_linguistic")

up_excerpt_box = st.text_area("Fragment transkrypcji (użyty tylko, gdy nie wgrano pliku .txt powyżej):",
                              value="", height=100, key="upload_excerpt")
up_excerpt = up_transcript_text or up_excerpt_box

if uploaded_reports and st.button("🔍 Oceń wgrane raporty", key="judge_uploaded"):
    parsed = []
    for f in uploaded_reports:
        try:
            content = f.getvalue().decode("utf-8")
        except:
            content = f.read().decode("utf-8", errors="ignore")
        got = _reports_from_uploaded_json(content, f.name)
        if got: parsed.extend(got)

    if parsed:
        reports = {}
        for label, rep in parsed:
            uniq = label
            i = 2
            while uniq in reports:
                uniq = f"{label} #{i}"
                i += 1
            reports[uniq] = rep

        with st.spinner("Sędzia ocenia wgrane raporty..."):
            up_txt_factual = read_golden_set(up_golden_factual_file)
            up_txt_linguistic = read_golden_set(up_golden_linguistic_file)

            up_engine = EvaluationEngine(LLMGateway(ObservabilityManager()), up_judge_model, **up_judge_cfg)
            up_eval, up_extra_metrics = asyncio.run(up_engine.evaluate(
                transcript_excerpt=up_excerpt, reports=reports, duration_sec=up_duration_sec,
                total_words=up_total_words, expected_factual=up_exp_factual, expected_linguistic=up_exp_linguistic,
                golden_factual=up_txt_factual, golden_linguistic=up_txt_linguistic
            ))

        up_h2h_wins = {name: 0 for name in reports.keys()}
        if up_eval.pairwise:
            for pref in up_eval.pairwise:
                if pref.winner in up_h2h_wins:
                    up_h2h_wins[pref.winner] += 1

        rows = []
        for se in up_eval.per_scenario:
            ext = up_extra_metrics.get(se.scenario_name, {})
            costs = ext.get("costs", {})

            recall_pct = ext.get("error_recall_pct", -1.0)
            recall_str = "Brak danych" if recall_pct < 0 else f"{recall_pct}%"
            crit_pct = ext.get("error_recall_critical_pct", -1.0)
            crit_str = "—" if crit_pct is None or crit_pct < 0 else f"{crit_pct}%"

            actual_model = ext.get("actual_hegemon_model", "") or costs.get("actual_hegemon_model", "")
            actual_short = actual_model.split("/")[-1] if actual_model else "—"
            fb_used = ext.get("fallback_used", False)
            model_cell = f"⚠️ {actual_short} (fallback)" if fb_used else actual_short

            missing = ext.get("missing_sections", [])
            miss_cell = "✓ komplet" if not missing else f"⚠️ brak: {', '.join(missing)}"
            lim_cell = "⚠️ TAK" if ext.get("lost_in_middle") else "—"

            rows.append({
                "Raport": se.scenario_name,
                "🤖 Model (faktyczny)": model_cell,
                "🧩 Typ": ext.get("scenario_kind", ""),
                "🏆 Jakość (0-50)": se.rubric_total,
                "🧾 Kompletność": miss_cell,
                "🕳️ Lost-in-middle": lim_cell,
                "🎯 Odchylenie (RMSE)": ext.get("rmse", 0.0),
                "🎯 Wykryte błędy (%)": recall_str,
                "🎯 CRIT/HIGH (%)": crit_str,
                "🔤 TPW (Narzut)": ext.get("tpw", 0.0),
                "📦 Gęst. Meryt. (zn/tok)": ext.get("factual_density", 0.0),
                "📦 Gęst. Ling. (zn/tok)": ext.get("linguistic_density", 0.0),
                "📦 Tokeny MAP": costs.get("prior_tokens_total", 0),
                "📦 Hegemon IN (max)": costs.get("hegemon_tokens_in", 0),
                "📦 Hegemon OUT": costs.get("hegemon_tokens_out", 0),
                "📉 Koszt MAP ($)": costs.get("map_total_usd", 0.0),
                "📈 Koszt Hegemona ($)": costs.get("reduce_usd", 0.0),
                "Osadzenie": se.rubric.groundedness,
                "Wygrane H2H": up_h2h_wins.get(se.scenario_name, 0),
            })

        df_up = pd.DataFrame(rows)
        if not df_up.empty:
            df_up = df_up.sort_values(by=["Wygrane H2H", "🏆 Jakość (0-50)", "🎯 Odchylenie (RMSE)"],
                                      ascending=[False, False, True])
            df_up = df_up.drop(columns=["Wygrane H2H"])
            st.dataframe(df_up, hide_index=True, use_container_width=True)

        st.subheader("🔎 Dowody sędziego i Analiza Błędów")
        for se in up_eval.per_scenario:
            with st.expander(f"{se.scenario_name} — Osadzenie {se.rubric.groundedness}/10"):
                st.markdown(f"**Uzasadnienie (Tabela wyłapanych błędów):**\n\n{se.rubric.justification}")
