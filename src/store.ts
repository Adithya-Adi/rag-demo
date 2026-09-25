import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";

export function getVectorStore() {
  return PGVectorStore.initialize(
    new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
    {
      postgresConnectionOptions: { connectionString: process.env.DATABASE_URL },
      tableName: "embeddings",
    }
  );
}
