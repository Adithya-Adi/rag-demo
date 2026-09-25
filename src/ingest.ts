import { readFile } from "node:fs/promises";
import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { getVectorStore } from "./store.js";

async function loadFile(path: string): Promise<Document[]> {
  if (path.toLowerCase().endsWith(".pdf")) {
    return new PDFLoader(path).load();
  }
  const text = await readFile(path, "utf-8");
  return [new Document({ pageContent: text, metadata: { source: path } })];
}

async function main() {
  const path = process.argv[2];
  if (!path) {
    console.log("Usage: npm run ingest -- <path-to-file>");
    process.exit(1);
  }

  // 1. Load
  const docs = await loadFile(path);
  console.log(`Loaded ${docs.length} page(s) from ${path}`);

  // 2. Chunk
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const chunks = await splitter.splitDocuments(docs);
  console.log(`Split into ${chunks.length} chunks`);

  // 3. Embed + store in pgvector
  const vectorStore = await getVectorStore();
  await vectorStore.addDocuments(chunks);
  await vectorStore.end();
  console.log("Stored chunks in Postgres. Done!");
}

main();
