import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";
import type { Document } from "@langchain/core/documents";
import { getVectorStore } from "./store.js";

const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}`
);

async function main() {
  const question = process.argv.slice(2).join(" ");
  if (!question) {
    console.log('Usage: npm run query -- "your question"');
    process.exit(1);
  }

  const vectorStore = await getVectorStore();

  // 1. Retrieve: top 3 chunks most similar to the question
  const retrieve = vectorStore.asRetriever({ k: 3 }).withConfig({ runName: "Retrieve" });
  const joinChunks = RunnableLambda.from((chunks: Document[]) =>
    chunks.map((c) => c.pageContent).join("\n\n---\n\n")
  );

  // 2. Augment: fill the prompt with context + question
  const augment = prompt.withConfig({ runName: "Augment" });

  // 3. Generate: send to the LLM
  const generate = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 }).withConfig({ runName: "Generate" });

  // One chain = one trace in LangSmith, with each step nested inside it
  const rag = RunnableSequence.from([
    { context: retrieve.pipe(joinChunks), question: new RunnablePassthrough() },
    augment,
    generate,
    new StringOutputParser(),
  ]).withConfig({ runName: "RAG" });

  const answer = await rag.invoke(question);
  console.log("\nAnswer:\n", answer);

  await vectorStore.end();
  await awaitAllCallbacks(); // make sure the trace is sent to LangSmith before the script exits
}

main();
