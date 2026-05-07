# First RAG

This project is a simple retrieval-augmented generation (RAG) setup built with:

- Wikipedia text collection scripts
- paragraph-based document chunking
- OpenAI embeddings
- Chroma vector database

## Project Files

- `txt.py`: downloads and cleans Wikipedia articles into `.txt` files
- `docs/`: stores the generated text documents
- `ingestion_pipeline.py`: loads `.txt` files, chunks them by paragraph, creates embeddings, and stores them in Chroma

## How It Works

1. Run `txt.py` to download Wikipedia content into the `docs/` folder.
2. Run `ingestion_pipeline.py` to:
   - load all `.txt` documents
   - split them into paragraph chunks
   - generate embeddings with `text-embedding-3-small`
   - save the vectors in `chroma_db/`

## Setup

Create a `.env` file in the project root with your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

```powershell
python txt.py
python ingestion_pipeline.py
```

## Notes

- `CHAT_MODEL = "gpt-5-mini"` is prepared for the next query/generation step of the RAG app.
- If OpenAI quota is unavailable, ingestion will fail during embedding generation.
