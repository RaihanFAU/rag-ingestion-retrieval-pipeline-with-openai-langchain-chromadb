import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv


VECTOR_DB_DIR = Path("chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
DEFAULT_K = 4


def load_api_key() -> str | None:
    """Load the OpenAI API key from env vars or a simple .env fallback parser."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAIApiKey")
    if api_key:
        return api_key

    env_path = Path(".env")
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key in {"OPENAI_API_KEY", "OpenAIApiKey"} and value:
            return value

    return None


def load_vector_store(persist_directory: Path):
    """Load an existing Chroma vector store."""
    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain-chroma'. Install it with: pip install langchain-chroma"
        ) from exc

    try:
        from langchain_openai import OpenAIEmbeddings
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain-openai'. Install it with: pip install langchain-openai"
        ) from exc

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector DB directory not found: {persist_directory.resolve()}. "
            "Run ingestion_pipeline.py first."
        )

    api_key = load_api_key()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or OpenAIApiKey in your .env file before retrieval.")

    embeddings = OpenAIEmbeddings(api_key=api_key, model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def retrieve_context(vector_store, query: str, k: int = DEFAULT_K):
    """Retrieve top-k relevant document chunks for the query."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        raise ValueError("No documents were retrieved from the vector store.")

    return docs


def build_prompt(query: str, docs) -> str:
    """Build a grounded prompt from retrieved context."""
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename", "unknown")
        paragraph_index = meta.get("paragraph_index", "?")
        context_blocks.append(
            f"[{i}] {filename} paragraph {paragraph_index}\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_blocks)

    return (
        "You are a helpful RAG assistant. Use only the context to answer. "
        "If the answer is not present, clearly say you do not know based on the provided documents.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Return a concise answer followed by a short Sources section with [number] references."
    )


def generate_answer(query: str, docs):
    """Generate an answer using the retrieved context."""
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain-openai'. Install it with: pip install langchain-openai"
        ) from exc

    api_key = load_api_key()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or OpenAIApiKey in your .env file before retrieval.")

    llm = ChatOpenAI(api_key=api_key, model=CHAT_MODEL, temperature=0)
    prompt = build_prompt(query, docs)
    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


def extract_source_numbers(answer: str, max_index: int) -> list[int]:
    """Extract unique [number] source citations from the model answer."""
    numbers = []
    for match in re.findall(r"\[(\d+)\]", answer):
        idx = int(match)
        if 1 <= idx <= max_index and idx not in numbers:
            numbers.append(idx)
    return numbers


def print_source_chunks(docs, source_numbers: list[int]) -> None:
    """Print full retrieved chunks for cited sources."""
    print("\nSource Chunks:\n")

    if not source_numbers:
        source_numbers = list(range(1, len(docs) + 1))

    for idx in source_numbers:
        doc = docs[idx - 1]
        meta = doc.metadata or {}
        filename = meta.get("filename", "unknown")
        paragraph_index = meta.get("paragraph_index", "?")
        source_path = meta.get("source", "unknown")

        print(f"[{idx}] {filename} paragraph {paragraph_index}")
        print(f"source: {source_path}")
        print(doc.page_content.strip())
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve context from Chroma and answer a query.")
    parser.add_argument("query", help="Question to ask the RAG pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("Starting retrieval pipeline...")
    vector_store = load_vector_store(VECTOR_DB_DIR)
    docs = retrieve_context(vector_store, args.query, k=args.k)
    answer = generate_answer(args.query, docs)

    print("\nAnswer:\n")
    print(answer)
    source_numbers = extract_source_numbers(answer, max_index=len(docs))
    print_source_chunks(docs, source_numbers)


if __name__ == "__main__":
    main()
