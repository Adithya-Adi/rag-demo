# RAG Demo (TypeScript)

Minimal Retrieval-Augmented Generation with OpenAI, LangChain.js and Postgres + pgvector.

```
src/ingest.ts : file -> chunks -> embeddings -> Postgres (pgvector)
src/query.ts  : question -> similar chunks -> prompt (augment) -> LLM answer
src/store.ts  : shared pgvector connection
```

## Setup

```bash
docker compose up -d        # starts Postgres with pgvector
npm install
copy .env.example .env      # then put your OpenAI key in .env  
```

## Run

```bash
npm run ingest -- data/your_file.pdf   # .pdf or any text file
npm run query -- "your question here"
```
