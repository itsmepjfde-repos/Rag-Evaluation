import json
import os
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from chromadb.errors import ChromaError
from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "hr_policy_chunks"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
TOP_K_DEFAULT = 5
MAX_FILTERS = 4

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

INTENT_SYSTEM_PROMPT = """You route HR policy questions for a retrieval system.
Use the conversation only to understand the latest user question and resolve
follow-up references. Select one or more category/sub_category pairs only from
the allowed list provided by the application. Never invent or alter labels.
Return a JSON object with exactly these fields:
{
  "is_relevant": true,
  "intent": "short name for the user's intent",
  "intent_summary": "brief user-facing explanation of the routing decision",
  "search_query": "standalone search query for the current question",
  "filters": [
    {"category": "approved category", "sub_category": "approved sub-category"}
  ],
  "response": ""
}
If the latest question is not about a topic covered by the approved labels, set
"is_relevant" to false, use an empty filters list and empty search_query, and
write a brief, polite response in "response" explaining that this assistant can
help with the HR policies in the approved list. Do not answer unrelated
questions. If it is relevant, set "is_relevant" to true, select 1-4 approved
labels, and leave "response" empty. The summary must be concise; do not provide
hidden chain-of-thought. If uncertain whether an HR question fits, choose the
closest applicable approved pair.
"""

ANSWER_SYSTEM_PROMPT = """You are an HR policy helpdesk assistant. Answer the latest
user question using only the supplied policy excerpts. If the excerpts do not
contain enough information, clearly say what is missing instead of guessing.
Do not invent policy terms, amounts, eligibility rules, or exceptions. Cite
supporting excerpts using their supplied chunk IDs, for example [chunk
hr-policy-chunk-000001]. Keep the answer clear and concise.
"""


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_collection() -> Any:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' is empty. "
            "Run embedding.py before starting the agent."
        )
    return collection


def get_groq_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


def recent_history(messages: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-limit:]
        if message["role"] in {"user", "assistant"}
    ]


def classify_question(
    client: OpenAI,
    question: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    allowed_labels = [
        {"category": category, "sub_category": sub_category}
        for category, sub_categories in ALLOWED_SUB_CATEGORIES.items()
        for sub_category in sorted(sub_categories)
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "allowed_labels": allowed_labels,
                        "recent_conversation": history,
                        "latest_question": question,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The intent model returned an empty response.")

    try:
        result: Any = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The intent model did not return valid JSON.") from error

    required_fields = {
        "is_relevant",
        "intent",
        "intent_summary",
        "search_query",
        "filters",
        "response",
    }
    if not isinstance(result, dict) or set(result) != required_fields:
        raise ValueError("The intent model returned an unexpected response format.")
    if not isinstance(result["is_relevant"], bool):
        raise ValueError("The intent model returned an invalid relevance decision.")
    if not all(
        isinstance(result[field], str)
        for field in ("intent", "intent_summary", "search_query", "response")
    ):
        raise ValueError("The intent model returned invalid intent details.")
    if not result["intent"].strip() or not result["intent_summary"].strip():
        raise ValueError("The intent model returned empty intent details.")

    filters = result["filters"]
    if not isinstance(filters, list):
        raise ValueError("The intent model returned invalid metadata filters.")

    if not result["is_relevant"]:
        if filters or result["search_query"].strip() or not result["response"].strip():
            raise ValueError(
                "The intent model returned an invalid out-of-scope response."
            )
        return result

    if not result["search_query"].strip() or result["response"].strip():
        raise ValueError("The intent model returned invalid in-scope intent details.")
    if not filters or len(filters) > MAX_FILTERS:
        raise ValueError(
            f"The intent model must select between 1 and {MAX_FILTERS} approved labels."
        )

    approved_filters: list[dict[str, str]] = []
    for item in filters:
        if not isinstance(item, dict) or set(item) != {"category", "sub_category"}:
            raise ValueError("The intent model returned a malformed metadata filter.")
        category = item["category"]
        sub_category = item["sub_category"]
        if (
            not isinstance(category, str)
            or not isinstance(sub_category, str)
            or sub_category not in ALLOWED_SUB_CATEGORIES.get(category, set())
        ):
            raise ValueError(
                "The intent model selected an unapproved category/sub-category."
            )
        label = {"category": category, "sub_category": sub_category}
        if label not in approved_filters:
            approved_filters.append(label)

    result["filters"] = approved_filters
    return result


def search_chunks(
    collection: Any,
    embedding_model: SentenceTransformer,
    search_query: str,
    filters: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = embedding_model.encode(
        [search_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()
    label_filters = [
        {
            "$and": [
                {"category": {"$eq": item["category"]}},
                {"sub_category": {"$eq": item["sub_category"]}},
            ]
        }
        for item in filters
    ]
    where_filter = (
        label_filters[0]
        if len(label_filters) == 1
        else {"$or": label_filters}
    )
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if document is None or metadata is None or distance is None:
            continue
        retrieved.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "similarity": 1.0 - float(distance),
            }
        )
    return retrieved


def generate_answer(
    client: OpenAI,
    question: str,
    history: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> str:
    excerpts = [
        {
            "chunk_id": chunk["id"],
            "metadata": chunk["metadata"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "latest_question": question,
                        "policy_excerpts": excerpts,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("The answer model returned an empty response.")
    return answer.strip()


def show_pipeline_details(
    trace: dict[str, Any],
    answer: str,
) -> None:
    with st.expander("Pipeline steps and decision summaries", expanded=True):
        st.markdown("**1. Question received**")
        st.write(trace["question"])

        st.markdown("**2. Intent and metadata routing**")
        st.write(f"**Intent:** {trace['intent']}")
        st.write(trace["intent_summary"])
        if trace["is_relevant"]:
            st.write(f"**Search query:** {trace['search_query']}")
            st.write("**Approved metadata filters:**")
            for item in trace["filters"]:
                st.write(
                    f"- `{item['category']}` / `{item['sub_category']}`"
                )
        else:
            st.info("No approved HR-policy category applies to this question.")

        st.markdown("**3. ChromaDB metadata + similarity search**")
        if not trace["is_relevant"]:
            st.write("Skipped because the question is outside the supported HR-policy topics.")
        elif not trace["chunks"]:
            st.info("No chunks matched the selected labels.")
        else:
            st.write(f"Retrieved {len(trace['chunks'])} matching chunk(s).")
            for chunk in trace["chunks"]:
                label = (
                    f"{chunk['metadata']['category']} / "
                    f"{chunk['metadata']['sub_category']}"
                )
                with st.expander(
                    f"{chunk['id']} | similarity {chunk['similarity']:.3f} | {label}"
                ):
                    st.write(chunk["text"])

        st.markdown("**4. Grounded answer**")
        if not trace["is_relevant"]:
            st.write("A polite scope clarification was generated by the intent model.")
        elif trace["chunks"]:
            st.write(
                "Generated from the retrieved policy excerpts shown above. "
                "The answer includes chunk-ID citations."
            )
        else:
            st.write("An answer was not generated because no chunks matched.")
        st.markdown(answer)


def main() -> None:
    st.set_page_config(page_title="HR Policy Helpdesk", page_icon="📘")
    st.title("HR Policy Helpdesk")
    st.caption(
        "Ask about leave, parental leave, remote work, or medical insurance policies."
    )

    with st.sidebar:
        st.subheader("Agent settings")
        st.write(f"Groq-compatible model: `{GROQ_MODEL}`")
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=TOP_K_DEFAULT)
        api_key = st.text_input(
            "Groq API key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Set GROQ_API_KEY in the environment or enter it here for this session.",
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("trace"):
                show_pipeline_details(message["trace"], message["content"])

    question = st.chat_input("Ask a question about the HR policies")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key.strip():
            error_message = (
                "A Groq API key is required. Set `GROQ_API_KEY` or enter it "
                "in the sidebar."
            )
            st.error(error_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_message}
            )
            return

        trace: dict[str, Any] = {
            "question": question,
            "is_relevant": False,
            "intent": "",
            "intent_summary": "",
            "search_query": "",
            "filters": [],
            "chunks": [],
        }
        answer = ""
        try:
            client = get_groq_client(api_key.strip())
            history = recent_history(st.session_state.chat_history[:-1])
            with st.status("Processing your question", expanded=True) as status:
                st.write("Step 1/4: Question received.")

                st.write("Step 2/4: Identifying intent and approved metadata labels.")
                intent = classify_question(client, question, history)
                trace.update(
                    is_relevant=intent["is_relevant"],
                    intent=intent["intent"],
                    intent_summary=intent["intent_summary"],
                    search_query=intent["search_query"],
                    filters=intent["filters"],
                )
                st.write(f"Intent: {trace['intent']}")
                st.write(trace["intent_summary"])
                if trace["is_relevant"]:
                    st.write(
                        "Filters: "
                        + ", ".join(
                            f"{item['category']} / {item['sub_category']}"
                            for item in trace["filters"]
                        )
                    )
                    st.write("Step 3/4: Searching matching ChromaDB chunks by metadata and similarity.")
                    collection = get_collection()
                    embedding_model = get_embedding_model()
                    trace["chunks"] = search_chunks(
                        collection,
                        embedding_model,
                        trace["search_query"],
                        trace["filters"],
                        top_k,
                    )
                    st.write(f"Retrieved {len(trace['chunks'])} matching chunk(s).")
                    for chunk in trace["chunks"]:
                        st.write(
                            f"{chunk['id']} — similarity {chunk['similarity']:.3f}"
                        )

                    if trace["chunks"]:
                        st.write("Step 4/4: Generating an answer grounded in retrieved excerpts.")
                        answer = generate_answer(client, question, history, trace["chunks"])
                    else:
                        answer = (
                            "I couldn't find matching policy excerpts for this question "
                            "under the selected metadata labels, so I can't provide a "
                            "policy-based answer."
                        )
                        st.write("Step 4/4: No answer generated because no chunks matched.")
                else:
                    answer = intent["response"]
                    st.write("Step 3/4: Skipped; no approved HR-policy category matched.")
                    st.write("Step 4/4: Showing the LLM's polite scope clarification.")
                status.update(label="Question processed", state="complete")

            st.markdown(answer)
            show_pipeline_details(trace, answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except (OpenAIError, ChromaError, ValueError, RuntimeError) as error:
            error_message = f"The agent could not complete this request: {error}"
            st.error(error_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_message}
            )


if __name__ == "__main__":
    main()