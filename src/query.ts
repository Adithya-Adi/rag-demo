import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
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

  // 1. Retrieve: top 3 chunks most similar to the question
  const vectorStore = await getVectorStore();
  const chunks = await vectorStore.similaritySearch(question, 3);
  await vectorStore.end();
  const context = chunks.map((c) => c.pageContent).join("\n\n---\n\n");

  // 2. Augment: fill the prompt with context + question
  const messages = await prompt.formatMessages({ context, question });

  // 3. Generate: send to the LLM
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const answer = await llm.invoke(messages);

  console.log("\nAnswer:\n", answer.content);
  console.log("\nSources:");
  for (const c of chunks) {
    console.log(" -", c.metadata.source, "page", c.metadata.loc?.pageNumber ?? "-");
  }
}

main();
