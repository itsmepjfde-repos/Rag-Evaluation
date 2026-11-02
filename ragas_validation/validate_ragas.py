import asyncio
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(WORKSPACE_ROOT / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
SYNTHETIC_DATASETS_DIR = Path(__file__).resolve().parent / "synthetic_datasets"
NO_CONTEXT_ANSWER = (
    "I couldn't find matching policy excerpts for this question under the "
    "selected metadata labels, so I can't provide a policy-based answer."
)


class AgentSentenceTransformerEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model: SentenceTransformer, model_name: str) -> None:
        super().__init__()
        self.model = model_name
        self.sentence_transformer = model
        self.set_run_config(RunConfig())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.sentence_transformer.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.sentence_transformer.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def load_test_cases(test_cases_path: Path) -> list[dict[str, str]]:
    if not test_cases_path.is_file():
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    with test_cases_path.open("r", encoding="utf-8") as test_cases_file:
        test_cases: Any = json.load(test_cases_file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("The test cases file must contain a non-empty JSON list.")

    required_fields = {
        "id",
        "category",
        "sub_category",
        "question",
        "ground_truth",
    }
    validated_cases: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict) or set(case) != required_fields:
            raise ValueError(
                f"Test case {index} must contain exactly: "
                f"{', '.join(sorted(required_fields))}."
            )
        if not all(isinstance(value, str) and value.strip() for value in case.values()):
            raise ValueError(f"Test case {index} contains an empty or invalid value.")
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate test case id: {case['id']}")
        seen_ids.add(case["id"])
        validated_cases.append(case)

    return validated_cases


def generate_synthetic_cases(testset_size: int) -> list[dict[str, str]]:
    if not 1 <= testset_size <= 50:
        raise ValueError("Synthetic testset size must be between 1 and 50.")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY was not found in the workspace-root .env or environment."
        )

    project_dir = Path(__file__).resolve().parent.parent
    chunks_path = project_dir / "hr_policy_chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Policy chunks file not found: {chunks_path}")

    from langchain_core.documents import Document
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    documents: list[Document] = []
    labels_by_text: dict[str, dict[str, str]] = {}
    with chunks_path.open("r", encoding="utf-8") as chunks_file:
        for line_number, line in enumerate(chunks_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {chunks_path}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record on line {line_number} of {chunks_path}")

            text = record.get("text")
            metadata = record.get("metadata")
            if (
                not isinstance(text, str)
                or not text.strip()
                or not isinstance(metadata, dict)
                or not isinstance(metadata.get("category"), str)
                or not isinstance(metadata.get("sub_category"), str)
            ):
                raise ValueError(
                    f"Invalid text or category metadata on line {line_number} "
                    f"of {chunks_path}"
                )
            label = {
                "category": metadata["category"],
                "sub_category": metadata["sub_category"],
            }
            if text in labels_by_text and labels_by_text[text] != label:
                raise ValueError(
                    "Duplicate policy text has conflicting category labels; "
                    "cannot reliably label generated questions."
                )
            labels_by_text[text] = label
            documents.append(Document(page_content=text, metadata=label))

    if not documents:
        raise ValueError(f"No policy chunks found in {chunks_path}.")

    sys.path.insert(0, str(project_dir))
    ragas_llm = llm_factory(
        model=GROQ_MODEL,
        provider="openai",
        client=OpenAI(api_key=api_key, base_url=GROQ_BASE_URL),
    )
    import HR_agent

    embedding_model = AgentSentenceTransformerEmbeddings(
        HR_agent.get_embedding_model(),
        HR_agent.EMBEDDING_MODEL,
    )
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=embedding_model,
        llm_context=(
            "Generate questions about the supplied company HR policy only. "
            "Use details explicitly stated in the policy and do not invent "
            "company rules or fill in placeholder amounts or dates."
        ),
    )
    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=testset_size,
        query_distribution=[
            (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0)
        ],
        raise_exceptions=True,
    )

    generated_cases: list[dict[str, str]] = []
    for index, sample in enumerate(testset.to_list(), start=1):
        question = sample.get("user_input")
        ground_truth = sample.get("reference")
        reference_contexts = sample.get("reference_contexts")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(ground_truth, str)
            or not ground_truth.strip()
            or not isinstance(reference_contexts, list)
            or not reference_contexts
        ):
            raise ValueError(
                f"Generated sample {index} is missing a question, answer, "
                "or reference context."
            )

        source_labels: set[tuple[str, str]] = set()
        for context in reference_contexts:
            if not isinstance(context, str) or context not in labels_by_text:
                raise ValueError(
                    f"Generated sample {index} refers to a context that cannot "
                    "be matched to a policy chunk."
                )
            label = labels_by_text[context]
            source_labels.add((label["category"], label["sub_category"]))
        if len(source_labels) != 1:
            raise ValueError(
                f"Generated sample {index} spans multiple policy categories; "
                "cannot assign one expected routing label."
            )

        category, sub_category = next(iter(source_labels))
        generated_cases.append(
            {
                "id": f"synthetic-{index:03d}",
                "category": category,
                "sub_category": sub_category,
                "question": question.strip(),
                "ground_truth": ground_truth.strip(),
            }
        )

    if len(generated_cases) != testset_size:
        raise ValueError(
            f"Ragas generated {len(generated_cases)} cases, "
            f"expected {testset_size}."
        )
    return generated_cases


def save_synthetic_dataset(
    dataset_name: str,
    test_cases: list[dict[str, str]],
) -> Path:
    if not dataset_name.startswith("synthetic_") or Path(dataset_name).name != dataset_name:
        raise ValueError("Invalid synthetic dataset name.")
    validated_cases = validate_test_cases(test_cases)
    SYNTHETIC_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = SYNTHETIC_DATASETS_DIR / f"{dataset_name}.json"
    if dataset_path.exists():
        raise FileExistsError(f"Synthetic dataset already exists: {dataset_path}")
    with dataset_path.open("w", encoding="utf-8") as dataset_file:
        json.dump(validated_cases, dataset_file, indent=2, ensure_ascii=False)
    return dataset_path


def validate_test_cases(test_cases: list[dict[str, str]]) -> list[dict[str, str]]:
    required_fields = {
        "id",
        "category",
        "sub_category",
        "question",
        "ground_truth",
    }
    if not test_cases:
        raise ValueError("A synthetic dataset must contain at least one test case.")
    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(test_cases, start=1):
        if not isinstance(case, dict) or set(case) != required_fields:
            raise ValueError(
                f"Test case {index} must contain exactly: "
                f"{', '.join(sorted(required_fields))}."
            )
        if not all(
            isinstance(value, str) and value.strip() for value in case.values()
        ):
            raise ValueError(f"Test case {index} contains an empty or invalid value.")
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate test case id: {case['id']}")
        seen_ids.add(case["id"])
        validated.append(case)
    return validated


def write_observations(path: Path, observations: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(observations, report_file, indent=2, ensure_ascii=False)


def evaluate_agent(
    test_cases: list[dict[str, str]],
    observations_path: Path,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise SystemExit(
            "GROQ_API_KEY was not found in the workspace-root .env or environment."
        )

    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir))
    import HR_agent

    client = HR_agent.get_groq_client(os.environ["GROQ_API_KEY"].strip())
    collection = HR_agent.get_collection()
    embedding_model = HR_agent.get_embedding_model()
    observations: list[dict[str, Any]] = []

    for index, case in enumerate(test_cases, start=1):
        if progress_callback is not None:
            progress_callback(
                "case_started",
                {"index": index, "total": len(test_cases), "test_case": case},
            )
        intent = HR_agent.classify_question(client, case["question"], [])
        if progress_callback is not None:
            progress_callback(
                "routing_complete",
                {
                    "index": index,
                    "total": len(test_cases),
                    "intent": intent,
                },
            )
        chunks: list[dict[str, Any]] = []
        if intent["is_relevant"]:
            chunks = HR_agent.search_chunks(
                collection,
                embedding_model,
                intent["search_query"],
                intent["filters"],
                HR_agent.TOP_K_DEFAULT,
            )
        if progress_callback is not None:
            progress_callback(
                "retrieval_complete",
                {
                    "index": index,
                    "total": len(test_cases),
                    "chunks": chunks,
                },
            )

        if chunks:
            answer = HR_agent.generate_answer(client, case["question"], [], chunks)
        elif intent["is_relevant"]:
            answer = NO_CONTEXT_ANSWER
        else:
            answer = intent["response"]

        expected_label = {
            "category": case["category"],
            "sub_category": case["sub_category"],
        }
        retrieved_contexts = [chunk["text"] for chunk in chunks]
        observations.append(
            {
                **case,
                "answer": answer,
                "contexts": retrieved_contexts,
                "retrieved_chunk_ids": [chunk["id"] for chunk in chunks],
                "selected_filters": intent["filters"],
                "routing_correct": expected_label in intent["filters"],
                "is_relevant": intent["is_relevant"],
            }
        )
        write_observations(observations_path, observations)
        if progress_callback is not None:
            progress_callback(
                "answer_complete",
                {
                    "index": len(observations),
                    "total": len(test_cases),
                    "observation": observations[-1],
                },
            )

    return observations


def mean_metric_scores(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    averages: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        values = [
            float(row[metric_name])
            for row in rows
            if row.get(metric_name) is not None
            and math.isfinite(float(row[metric_name]))
        ]
        averages[metric_name] = sum(values) / len(values) if values else None
    return averages


def save_reports(
    run_dir: Path,
    observations: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    dataset_name: str = "curated",
) -> Path:
    if len(observations) != len(score_rows):
        raise ValueError("Ragas returned a different number of rows than expected.")

    report_rows = [
        {**observation, **scores}
        for observation, scores in zip(observations, score_rows)
    ]
    csv_fields = [
        "id",
        "category",
        "sub_category",
        "question",
        "ground_truth",
        "answer",
        "routing_correct",
        "is_relevant",
        "selected_filters",
        "retrieved_chunk_ids",
        *METRIC_NAMES,
    ]
    with (run_dir / "scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as report_file:
        writer = csv.DictWriter(report_file, fieldnames=csv_fields)
        writer.writeheader()
        for row in report_rows:
            csv_row = {field: row.get(field) for field in csv_fields}
            csv_row["selected_filters"] = json.dumps(
                csv_row["selected_filters"], ensure_ascii=False
            )
            csv_row["retrieved_chunk_ids"] = json.dumps(
                csv_row["retrieved_chunk_ids"], ensure_ascii=False
            )
            writer.writerow(csv_row)

    category_scores: dict[str, dict[str, float | None]] = {}
    for category in sorted({row["category"] for row in report_rows}):
        category_rows = [row for row in report_rows if row["category"] == category]
        category_scores[category] = mean_metric_scores(category_rows)

    summary = {
        "model": GROQ_MODEL,
        "dataset_name": dataset_name,
        "test_count": len(report_rows),
        "routing_accuracy": sum(bool(row["routing_correct"]) for row in report_rows)
        / len(report_rows),
        "overall_mean_scores": mean_metric_scores(report_rows),
        "mean_scores_by_category": category_scores,
        "files": ["observations.json", "scores.csv"],
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, indent=2, ensure_ascii=False)

    return run_dir


def run_validation(
    test_cases: list[dict[str, str]] | None = None,
    dataset_name: str = "curated",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> Path:
    validation_dir = Path(__file__).resolve().parent
    if test_cases is None:
        test_cases = load_test_cases(validation_dir / "test_cases.json")
    else:
        test_cases = validate_test_cases(test_cases)
    run_dir = (
        validation_dir
        / "results"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    if progress_callback is not None:
        progress_callback("run_started", {"run_dir": str(run_dir)})
    observations = evaluate_agent(
        test_cases,
        run_dir / "observations.json",
        progress_callback=progress_callback,
    )

    if progress_callback is not None:
        progress_callback(
            "scoring_started",
            {"test_count": len(observations), "metrics": list(METRIC_NAMES)},
        )
    ragas_llm = llm_factory(
        model=GROQ_MODEL,
        provider="openai",
        client=OpenAI(
            api_key=os.environ["GROQ_API_KEY"].strip(),
            base_url=GROQ_BASE_URL,
        ),
    )

    import HR_agent

    embeddings = AgentSentenceTransformerEmbeddings(
        HR_agent.get_embedding_model(),
        HR_agent.EMBEDDING_MODEL,
    )
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in observations
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        llm=ragas_llm,
        raise_exceptions=True,
        show_progress=True,
    )
    report_dir = save_reports(
        run_dir,
        observations,
        result.scores,
        dataset_name=dataset_name,
    )
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    if progress_callback is not None:
        progress_callback("scoring_complete", {"summary": summary})

    return report_dir


def main() -> None:
    report_dir = run_validation()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Ragas validation completed for {summary['test_count']} questions.")
    print(f"Routing accuracy: {summary['routing_accuracy']:.1%}")
    print(f"Mean scores: {summary['overall_mean_scores']}")
    print(f"Reports: {report_dir}")


if __name__ == "__main__":
    main()
