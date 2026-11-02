import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "hr_policy_chunks"
BATCH_SIZE = 100

ALLOWED_SUB_CATEGORIES = {
    "General HR Policy": {
        "Document Overview",
        "Employee Acknowledgment",
    },
    "Leave Policy": {
        "Leave Categories & Accrual",
        "General Leave Rules",
        "Public Holidays",
    },
    "Parental Leave Policy": {
        "Maternity Leave",
        "Adoption & Surrogacy",
        "Paternity Leave",
    },
    "Remote Work Policy": {
        "Hybrid Work Schedule",
        "Equipment & Home Office Allowance",
        "Security & Data Privacy",
    },
    "Medical Insurance Policy": {
        "Coverage & Dependents",
        "Enrollment & Claims",
    },
}


def load_chunks(jsonl_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue

            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {jsonl_path}"
                ) from error

            if not isinstance(record, dict) or set(record) != {"text", "metadata"}:
                raise ValueError(
                    f"Line {line_number} must contain only 'text' and 'metadata'."
                )

            text = record["text"]
            metadata = record["metadata"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number} has empty or invalid text.")
            if not isinstance(metadata, dict) or set(metadata) != {
                "category",
                "sub_category",
            }:
                raise ValueError(
                    f"Line {line_number} metadata must contain exactly "
                    "'category' and 'sub_category'."
                )

            category = metadata["category"]
            sub_category = metadata["sub_category"]
            if (
                not isinstance(category, str)
                or not isinstance(sub_category, str)
                or sub_category not in ALLOWED_SUB_CATEGORIES.get(category, set())
            ):
                raise ValueError(
                    f"Line {line_number} has an unsupported category/sub-category "
                    "combination."
                )

            documents.append(text)
            metadatas.append(
                {"category": category, "sub_category": sub_category}
            )

    if not documents:
        raise ValueError(f"No chunks found in {jsonl_path}.")

    return documents, metadatas


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    jsonl_path = project_dir / "hr_policy_chunks.jsonl"
    chroma_path = project_dir.parent / "chroma_db"

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Chunk JSONL file not found: {jsonl_path}")

    documents, metadatas = load_chunks(jsonl_path)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
        },
    )

    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        embeddings = model.encode(
            documents[start:end],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=end == len(documents),
        )
        collection.upsert(
            ids=[f"hr-policy-chunk-{index:06d}" for index in range(start, end)],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings.tolist(),
        )

    print(
        f"Stored {len(documents)} embedded chunks in collection "
        f"'{COLLECTION_NAME}' at '{chroma_path}'."
    )


if __name__ == "__main__":
    main()