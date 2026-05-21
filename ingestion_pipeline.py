import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv


DOCS_DIR = Path("docs")
VECTOR_DB_DIR = Path("chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"


def chunk_text_by_paragraph(text: str) -> list[str]:
    """Split text into paragraph chunks using blank lines."""
    return [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]


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


def load_text_documents(docs_dir: Path):
    """Load all .txt files and convert each paragraph into a LangChain Document."""
    try:
        from langchain_core.documents import Document
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain-core'. Install it with: pip install langchain-core"
        ) from exc

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir.resolve()}")

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in: {docs_dir.resolve()}")

    documents = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        paragraphs = chunk_text_by_paragraph(text)

        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            documents.append(
                Document(
                    page_content=paragraph,
                    metadata={
                        "source": str(txt_file.resolve()),
                        "filename": txt_file.name,
                        "paragraph_index": paragraph_index,
                    },
                )
            )

    if not documents:
        raise ValueError("Text files were found, but no paragraph chunks were created.")

    return documents


def build_vector_store(documents, persist_directory: Path):
    """Create embeddings and store the documents in Chroma."""
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

    api_key = load_api_key()
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY or OpenAIApiKey in your .env file before running ingestion."
        )

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=EMBEDDING_MODEL,
    )

    # Rebuild the vector DB from scratch on each run to avoid duplicate chunks.
    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return vector_store


def main() -> None:
    print("Starting the ingestion pipeline...")
    documents = load_text_documents(DOCS_DIR)
    build_vector_store(documents, VECTOR_DB_DIR)

    print(
        f"Ingestion complete. Loaded {len(documents)} paragraph chunks from "
        f"{DOCS_DIR.resolve()} into {VECTOR_DB_DIR.resolve()} using {EMBEDDING_MODEL}."
    )


if __name__ == "__main__":
    main()
