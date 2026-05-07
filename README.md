# First RAG

This project is a simple retrieval-augmented generation (RAG) setup built with:

- Wikipedia scraping with `requests` and `BeautifulSoup` (`bs4`)
- Wikipedia text collection scripts
- paragraph-based document chunking
- OpenAI embeddings
- Chroma vector database

## Tools And Packages

- `requests`: downloads Wikipedia HTML pages
- `beautifulsoup4` (`bs4`): parses and cleans article content from HTML
- `python-dotenv`: loads environment variables from `.env`
- `langchain-core`: creates document objects for chunked text
- `langchain-openai`: generates embeddings with OpenAI
- `langchain-chroma`: stores and loads vectors in Chroma
- `chromadb`: local vector database backend

## Project Files

- `txt.py`: downloads and cleans Wikipedia articles into `.txt` files
- `docs/`: stores the generated text documents
- `ingestion_pipeline.py`: loads `.txt` files, chunks them by paragraph, creates embeddings, and stores them in Chroma

## How Data Is Collected

1. `requests` downloads Wikipedia article pages.
2. `BeautifulSoup` (`bs4`) extracts clean article text from the HTML.
3. The cleaned content is saved as `.txt` files in the `docs/` folder.

## How It Works

1. Run `txt.py` to scrape and download Wikipedia content into the `docs/` folder.
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
