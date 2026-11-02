import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_ragas import (
    GROQ_MODEL,
    METRIC_NAMES,
    SYNTHETIC_DATASETS_DIR,
    generate_synthetic_cases,
    load_test_cases,
    run_validation,
    save_synthetic_dataset,
    validate_test_cases,
)


VALIDATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = VALIDATION_DIR / "results"
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


def available_runs() -> list[Path]:
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(
        (path for path in RESULTS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def available_synthetic_datasets() -> list[Path]:
    if not SYNTHETIC_DATASETS_DIR.is_dir():
        return []
    return sorted(
        SYNTHETIC_DATASETS_DIR.glob("synthetic_*.json"),
        key=lambda path: path.name,
        reverse=True,
    )


def new_synthetic_dataset_name() -> str:
    return f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def normalized_editor_cases(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required_fields = [
        "id",
        "category",
        "sub_category",
        "question",
        "ground_truth",
    ]
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        case = {field: row.get(field) for field in required_fields}
        case["evaluate"] = bool(row.get("evaluate", False))
        rows.append(case)
    return rows


def cases_for_storage(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return validate_test_cases(
        [
            {
                field: (
                    row[field].strip()
                    if isinstance(row.get(field), str)
                    else ""
                )
                for field in (
                    "id",
                    "category",
                    "sub_category",
                    "question",
                    "ground_truth",
                )
            }
            for row in rows
        ]
    )


def render_synthetic_panel() -> None:
    st.subheader("Generate and review synthetic test cases")
    st.write(
        "Ragas drafts questions and reference answers from the policy chunks. "
        "Review and edit every draft before saving it or using it for evaluation."
    )
    st.warning(
        "Generation uses Groq and can incur API usage. Generated answers are "
        "draft references, not verified policy facts."
    )

    saved_datasets = available_synthetic_datasets()
    options = [""] + [path.stem for path in saved_datasets]
    selected_saved_dataset = st.selectbox(
        "Load a saved synthetic dataset",
        options,
        format_func=lambda value: value or "Select a dataset",
        key="synthetic_saved_dataset_selector",
    )

    draft_rows = st.session_state.get("synthetic_draft_cases")
    draft_exists = isinstance(draft_rows, list) and bool(draft_rows)
    if st.button(
        "Load selected dataset",
        disabled=not selected_saved_dataset or draft_exists,
        key="load_synthetic_dataset",
    ):
        dataset_path = SYNTHETIC_DATASETS_DIR / f"{selected_saved_dataset}.json"
        loaded_cases = load_test_cases(dataset_path)
        st.session_state.synthetic_draft_cases = [
            {**case, "evaluate": True} for case in loaded_cases
        ]
        st.session_state.synthetic_draft_name = selected_saved_dataset
        st.session_state.synthetic_draft_source = selected_saved_dataset
        st.session_state.synthetic_draft_saved_path = str(dataset_path)
        st.session_state.synthetic_draft_saved_snapshot = loaded_cases
        st.session_state.synthetic_editor_revision = (
            st.session_state.get("synthetic_editor_revision", 0) + 1
        )
        st.rerun()

    size = st.number_input(
        "How many synthetic questions should Ragas create?",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )
    if st.button(
        "Generate synthetic draft",
        type="primary",
        disabled=draft_exists,
        key="generate_synthetic_dataset",
    ):
        try:
            with st.spinner(
                f"Generating {int(size)} questions from the HR policy chunks..."
            ):
                generated_cases = generate_synthetic_cases(int(size))
            st.session_state.synthetic_draft_cases = [
                {**case, "evaluate": True} for case in generated_cases
            ]
            st.session_state.synthetic_draft_name = new_synthetic_dataset_name()
            st.session_state.synthetic_draft_source = "generated"
            st.session_state.synthetic_draft_saved_path = None
            st.session_state.synthetic_draft_saved_snapshot = None
            st.session_state.synthetic_editor_revision = (
                st.session_state.get("synthetic_editor_revision", 0) + 1
            )
            st.success(
                "Synthetic draft generated. Review the questions and reference "
                "answers below before saving."
            )
            st.rerun()
        except Exception as error:
            st.error(f"Could not generate the synthetic dataset: {error}")
            st.exception(error)

    draft_rows = st.session_state.get("synthetic_draft_cases")
    if not isinstance(draft_rows, list) or not draft_rows:
        if draft_exists:
            st.info(
                "Save or discard the current draft before generating or loading "
                "another dataset."
            )
        else:
            st.info("Generate a draft or load a previously saved synthetic dataset.")
        return

    dataset_name = st.session_state.synthetic_draft_name
    revision = st.session_state.get("synthetic_editor_revision", 0)
    st.markdown(f"**Draft:** `{dataset_name}`")
    st.caption(
        "Edit question/reference text, and check only the rows you want to "
        "include in the next evaluation. Category labels are fixed to the "
        "source policy chunk."
    )
    select_all_col, select_none_col = st.columns(2)
    with select_all_col:
        select_all_clicked = st.button(
            "Select all questions",
            key=f"select_all_{dataset_name}_{revision}",
        )
    with select_none_col:
        select_none_clicked = st.button(
            "Deselect all questions",
            key=f"select_none_{dataset_name}_{revision}",
        )
    editor_key = f"synthetic_case_editor_{dataset_name}_{revision}"
    edited_frame = st.data_editor(
        pd.DataFrame(draft_rows),
        key=editor_key,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["id", "category", "sub_category"],
        column_config={
            "evaluate": st.column_config.CheckboxColumn(
                "Evaluate",
                help="Include this case in the next Ragas run.",
                default=True,
            ),
            "id": st.column_config.TextColumn("ID"),
            "category": st.column_config.TextColumn("Category"),
            "sub_category": st.column_config.TextColumn("Subcategory"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth": st.column_config.TextColumn(
                "Reference answer", width="large"
            ),
        },
    )
    current_rows = normalized_editor_cases(edited_frame)
    if select_all_clicked or select_none_clicked:
        selected = select_all_clicked
        for row in current_rows:
            row["evaluate"] = selected
        revision += 1
        st.session_state.synthetic_editor_revision = revision
    st.session_state.synthetic_draft_cases = current_rows
    if select_all_clicked or select_none_clicked:
        st.rerun()
    try:
        current_cases = cases_for_storage(current_rows)
        validation_error = None
    except (TypeError, ValueError, KeyError) as error:
        current_cases = []
        validation_error = str(error)

    saved_path_value = st.session_state.get("synthetic_draft_saved_path")
    saved_snapshot = st.session_state.get("synthetic_draft_saved_snapshot")
    has_saved_copy = isinstance(saved_path_value, str) and bool(saved_path_value)
    dirty = has_saved_copy and current_cases != saved_snapshot
    if validation_error:
        st.error(f"Fix the draft before saving or evaluating: {validation_error}")
    elif dirty:
        st.warning("The saved dataset differs from your current edits. Save changes before evaluation.")

    selected_cases = [
        case
        for case, row in zip(current_cases, current_rows)
        if row.get("evaluate", False)
    ]
    st.caption(
        f"{len(current_rows)} reviewed draft case(s); "
        f"{len(selected_cases)} selected for evaluation."
    )

    save_col, discard_col, run_col = st.columns([1, 1, 2])
    with save_col:
        save_clicked = st.button(
            "Save reviewed dataset" if dirty or not has_saved_copy else "Dataset saved",
            disabled=bool(validation_error) or (has_saved_copy and not dirty),
            key=f"save_{dataset_name}_{revision}",
        )
    with discard_col:
        discard_clicked = st.button(
            "Discard draft",
            key=f"discard_{dataset_name}_{revision}",
        )
    with run_col:
        run_clicked = st.button(
            "Evaluate selected questions",
            type="primary",
            disabled=(
                not has_saved_copy
                or dirty
                or bool(validation_error)
                or not selected_cases
            ),
            key=f"evaluate_{dataset_name}_{revision}",
        )

    if discard_clicked:
        for key in (
            "synthetic_draft_cases",
            "synthetic_draft_name",
            "synthetic_draft_source",
            "synthetic_draft_saved_path",
            "synthetic_draft_saved_snapshot",
        ):
            st.session_state.pop(key, None)
        st.session_state.synthetic_editor_revision = revision + 1
        st.rerun()

    if save_clicked:
        try:
            saved_dataset_name = (
                new_synthetic_dataset_name()
                if has_saved_copy and dirty
                else dataset_name
            )
            path = save_synthetic_dataset(saved_dataset_name, current_cases)
            st.session_state.synthetic_draft_name = saved_dataset_name
            st.session_state.synthetic_draft_saved_path = str(path)
            st.session_state.synthetic_draft_saved_snapshot = current_cases
            if saved_dataset_name != dataset_name:
                st.session_state.synthetic_editor_revision = revision + 1
            st.success(f"Saved reviewed dataset: `{path}`")
        except (OSError, TypeError, ValueError) as error:
            st.error(f"Could not save the synthetic dataset: {error}")

    if run_clicked:
        if not has_saved_copy or dirty:
            st.error("Save your reviewed edits before evaluation.")
            return
        progress = st.progress(0, text="Starting selected-case evaluation...")
        status = st.empty()

        def update_selected_run(event: str, payload: dict[str, Any]) -> None:
            if event == "case_started":
                status.info(
                    f"Test {payload['index']}/{payload['total']}: "
                    f"{payload['test_case']['question']}"
                )
            elif event == "answer_complete":
                progress.progress(
                    payload["index"] / payload["total"],
                    text=f"Agent workflow: {payload['index']}/{payload['total']}",
                )
            elif event == "scoring_started":
                status.warning("Agent answers ready; Ragas scoring is in progress.")
            elif event == "scoring_complete":
                status.success("Selected-case evaluation completed.")

        try:
            report_dir = run_validation(
                test_cases=selected_cases,
                dataset_name=f"{dataset_name} ({len(selected_cases)} selected)",
                progress_callback=update_selected_run,
            )
            st.session_state.selected_run = str(report_dir)
            st.success(f"Saved evaluation results to `{report_dir}`")
            st.rerun()
        except Exception as error:
            st.error(f"Selected-case evaluation failed: {error}")
            st.exception(error)
def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as data_file:
        value: Any = json.load(data_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    scores_path = run_dir / "scores.csv"
    if not summary_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(f"This run is incomplete: {run_dir}")
    summary = read_json(summary_path)
    scores = pd.read_csv(scores_path).to_dict(orient="records")
    return summary, scores


def render_score_cards(summary: dict[str, Any]) -> None:
    metrics = summary.get("overall_mean_scores", {})
    columns = st.columns(6)
    columns[0].metric("Questions", summary.get("test_count", 0))
    routing = summary.get("routing_accuracy")
    columns[1].metric(
        "Routing accuracy",
        f"{routing:.1%}" if isinstance(routing, (float, int)) else "N/A",
    )
    for column, metric_name in zip(columns[2:], METRIC_NAMES):
        value = metrics.get(metric_name)
        column.metric(
            METRIC_LABELS[metric_name],
            f"{value:.3f}" if isinstance(value, (float, int)) else "N/A",
        )


def render_overview(
    summary: dict[str, Any],
    scores: list[dict[str, Any]],
) -> None:
    render_score_cards(summary)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Mean Ragas scores by policy category")
        categories = summary.get("mean_scores_by_category", {})
        if categories:
            category_frame = pd.DataFrame.from_dict(categories, orient="index")
            category_frame = category_frame.rename(columns=METRIC_LABELS)
            st.bar_chart(category_frame)
        else:
            st.info("This run has no category-level scores.")
    with right:
        st.subheader("Routing results")
        if scores:
            correct = sum(bool(row.get("routing_correct")) for row in scores)
            st.write(f"{correct} of {len(scores)} questions routed to the expected category.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Question ID": row.get("id"),
                            "Category": row.get("category"),
                            "Expected route selected": bool(
                                row.get("routing_correct")
                            ),
                        }
                        for row in scores
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    st.subheader("Per-question Ragas scores")
    score_columns = [
        "id",
        "category",
        "sub_category",
        "routing_correct",
        *METRIC_NAMES,
    ]
    frame = pd.DataFrame(scores)
    available_columns = [column for column in score_columns if column in frame]
    if available_columns:
        st.dataframe(
            frame[available_columns],
            hide_index=True,
            width="stretch",
        )


def render_case_details(scores: list[dict[str, Any]], run_dir: Path) -> None:
    observations_path = run_dir / "observations.json"
    if observations_path.is_file():
        with observations_path.open("r", encoding="utf-8") as observations_file:
            observations: Any = json.load(observations_file)
    else:
        observations = []
    if not isinstance(observations, list) or not observations:
        st.info("No saved agent observations are available for this run.")
        return

    observation_by_id = {
        row["id"]: row
        for row in observations
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    labels = [
        f"{row.get('id', 'unknown')} — {row.get('question', '')}"
        for row in observations
        if isinstance(row, dict)
    ]
    selected_label = st.selectbox("Select a test question", labels)
    selected_index = labels.index(selected_label)
    observation = observations[selected_index]
    score = next(
        (row for row in scores if row.get("id") == observation.get("id")),
        observation_by_id.get(observation.get("id"), {}),
    )

    st.markdown(f"**Category:** {observation.get('category')} / {observation.get('sub_category')}")
    st.markdown("**Input question**")
    st.info(observation.get("question", ""))
    st.markdown("**Reference answer**")
    st.write(observation.get("ground_truth", ""))

    st.markdown("**Agent routing output**")
    routing_col, filters_col = st.columns([1, 2])
    routing_col.metric(
        "Expected category selected",
        "Yes" if observation.get("routing_correct") else "No",
    )
    filters_col.json(observation.get("selected_filters", []))

    st.markdown("**Retrieved chunks**")
    chunk_ids = observation.get("retrieved_chunk_ids", [])
    contexts = observation.get("contexts", [])
    if not contexts:
        st.warning("The agent retrieved no context for this question.")
    for index, context in enumerate(contexts):
        chunk_id = chunk_ids[index] if index < len(chunk_ids) else f"chunk {index + 1}"
        with st.expander(f"{chunk_id}"):
            st.write(context)

    st.markdown("**Generated agent answer**")
    st.write(observation.get("answer", ""))
    st.markdown("**Ragas scores**")
    metric_columns = st.columns(4)
    for column, metric_name in zip(metric_columns, METRIC_NAMES):
        value = score.get(metric_name)
        if isinstance(value, (float, int)) and pd.notna(value):
            column.metric(METRIC_LABELS[metric_name], f"{value:.3f}")
        else:
            column.metric(METRIC_LABELS[metric_name], "N/A")


def render_run_history(runs: list[Path], current_run: Path) -> None:
    history_rows: list[dict[str, Any]] = []
    for run_dir in reversed(runs):
        try:
            summary, _ = read_run(run_dir)
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            continue
        history_rows.append(
            {
                "run": run_dir.name,
                "routing_accuracy": summary.get("routing_accuracy"),
                **summary.get("overall_mean_scores", {}),
            }
        )
    if len(history_rows) < 2:
        st.info("Run another evaluation to compare results over time.")
        return

    frame = pd.DataFrame(history_rows).set_index("run")
    frame = frame.rename(columns=METRIC_LABELS)
    st.line_chart(frame)
    st.caption(f"Selected run: {current_run.name}")


def main() -> None:
    st.set_page_config(
        page_title="HR Helpdesk Ragas Evaluation",
        page_icon="📊",
        layout="wide",
    )
    st.title("HR Helpdesk · Ragas Evaluation")
    st.caption(
        "Inspect the evaluation workflow, agent inputs and outputs, and Ragas "
        "scores across test runs."
    )
    st.warning(
        "Starting an evaluation calls Groq for agent answers and Ragas judging. "
        "It may take several minutes and incur API usage."
    )

    runs = available_runs()
    if "selected_run" not in st.session_state:
        st.session_state.selected_run = str(runs[0]) if runs else ""

    with st.sidebar:
        st.subheader("Evaluation controls")
        st.write(f"Model: `{GROQ_MODEL}`")
        run_button = st.button(
            "Run full validation",
            type="primary",
            width="stretch",
        )
        if runs:
            run_paths = [str(path) for path in runs]
            default_index = (
                run_paths.index(st.session_state.selected_run)
                if st.session_state.selected_run in run_paths
                else 0
            )
            chosen_run = st.selectbox(
                "View saved run",
                run_paths,
                index=default_index,
                format_func=lambda value: Path(value).name,
            )
            st.session_state.selected_run = chosen_run
        else:
            st.info("No saved evaluation runs yet.")

    if run_button:
        progress = st.progress(0, text="Preparing validation...")
        workflow = st.container()
        current_status = st.empty()
        active_case: dict[str, Any] = {}
        case_panel = st.empty()

        def update_workflow(event: str, payload: dict[str, Any]) -> None:
            if event == "run_started":
                current_status.info(f"Run created: `{payload['run_dir']}`")
            elif event == "case_started":
                active_case.clear()
                active_case.update(payload)
                current_status.info(
                    f"Test {payload['index']}/{payload['total']}: "
                    f"{payload['test_case']['question']}"
                )
                with workflow:
                    st.markdown(
                        f"**Input · {payload['test_case']['id']}**  \n"
                        f"{payload['test_case']['question']}"
                    )
            elif event == "routing_complete":
                intent = payload["intent"]
                with workflow:
                    st.write(
                        "**Routing output:** "
                        + json.dumps(intent.get("filters", []), ensure_ascii=False)
                    )
            elif event == "retrieval_complete":
                chunks = payload["chunks"]
                with workflow:
                    st.write(
                        f"**Retrieval output:** {len(chunks)} chunk(s) — "
                        + ", ".join(chunk["id"] for chunk in chunks)
                    )
            elif event == "answer_complete":
                observation = payload["observation"]
                progress.progress(
                    payload["index"] / payload["total"],
                    text=f"Agent workflow: {payload['index']}/{payload['total']}",
                )
                with workflow:
                    st.write("**Generated answer:** " + observation["answer"])
                    st.divider()
                case_panel.json(
                    {
                        "latest_case": observation["id"],
                        "routing_correct": observation["routing_correct"],
                        "retrieved_chunk_ids": observation["retrieved_chunk_ids"],
                    }
                )
            elif event == "scoring_started":
                progress.progress(1.0, text="Agent workflow complete; Ragas scoring in progress...")
                current_status.warning(
                    f"Scoring {payload['test_count']} answers with: "
                    + ", ".join(payload["metrics"])
                )
            elif event == "scoring_complete":
                current_status.success("Evaluation and scoring completed.")

        try:
            report_dir = run_validation(
                dataset_name="curated",
                progress_callback=update_workflow,
            )
            st.session_state.selected_run = str(report_dir)
            st.success(f"Saved results to `{report_dir}`")
            st.rerun()
        except Exception as error:
            st.error(f"Evaluation failed: {error}")
            st.exception(error)
            st.info(
                "Any per-question observations completed before the failure are "
                "saved in the run's `observations.json`."
            )

    overview_tab, cases_tab, history_tab, workflow_tab, synthetic_tab = st.tabs(
        [
            "Dashboard",
            "Question details",
            "Run comparison",
            "Workflow",
            "Synthetic test data",
        ]
    )
    selected_run = (
        Path(st.session_state.selected_run)
        if st.session_state.selected_run
        else None
    )
    summary: dict[str, Any] | None = None
    scores: list[dict[str, Any]] = []
    run_has_observations = False
    if selected_run is not None:
        run_has_observations = (selected_run / "observations.json").is_file()
        try:
            if (selected_run / "summary.json").is_file():
                summary, scores = read_run(selected_run)
            else:
                st.warning(f"This run has no completed Ragas summary: `{selected_run}`")
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as error:
            st.error(f"Could not load this run: {error}")

    with overview_tab:
        if summary is None:
            st.info(
                "Run validation or select a completed results folder to view "
                "the consolidated dashboard."
            )
        else:
            render_overview(summary, scores)
    with cases_tab:
        if summary is None or selected_run is None:
            st.info("Question details are available after an evaluation run.")
        else:
            render_case_details(scores, selected_run)
    with history_tab:
        if selected_run is None:
            st.info("Run history is available after an evaluation run.")
        else:
            render_run_history(runs, selected_run)
    with workflow_tab:
        if summary is None:
            st.info("Workflow details are available after an evaluation run.")
        else:
            st.write(
                "Each test follows this path: **question + reference answer → "
                "intent routing → metadata-filtered Chroma retrieval → Groq answer → "
                "Ragas scoring**."
            )
            st.write("Ragas metrics: " + ", ".join(METRIC_LABELS[name] for name in METRIC_NAMES))
            st.write(f"Dataset: `{summary.get('dataset_name', 'curated')}`")
            st.write(f"Agent model: `{summary.get('model', 'unknown')}`")
            if run_has_observations and selected_run is not None:
                observations_path = selected_run / "observations.json"
                with observations_path.open("r", encoding="utf-8") as observations_file:
                    observations: Any = json.load(observations_file)
                if isinstance(observations, list):
                    st.write(f"Saved workflow records: {len(observations)}")
                    st.json(observations)
    with synthetic_tab:
        render_synthetic_panel()


if __name__ == "__main__":
    main()
