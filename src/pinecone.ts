/**
 * Pinecone Vector Store Module
 * Handles index setup and chunk storage using Ollama embeddings.
 */

import { Pinecone } from "@pinecone-database/pinecone";
import { getEmbedding } from "./embedder";
import * as dotenv from "dotenv";
dotenv.config();

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
const INDEX_NAME = "rag-semantic-ts";

/**
 * Ensure the Pinecone index exists. Creates it if it doesn't.
 * Uses dimension 768 to match nomic-embed-text output.
 */
export async function setupPinecone() {
  const existing = await pc.listIndexes();
  const exists = existing.indexes?.some((i) => i.name === INDEX_NAME);

  if (!exists) {
    await pc.createIndex({
      name: INDEX_NAME,
      dimension: 768, // nomic-embed-text dimension is 768
      metric: "cosine",
      spec: {
        serverless: { cloud: "aws", region: "us-east-1" },
      },
    });
    console.log("✅ Index created!");
  } else {
    console.log("ℹ️  Index already exists, skipping creation.");
  }

  return pc.index(INDEX_NAME);
}

/**
 * Embed and store text chunks into Pinecone.
 * Each chunk is embedded via Ollama nomic-embed-text and upserted as a vector.
 */
export async function storeChunks(chunks: string[]) {
  const index = await setupPinecone();

  // Clear existing vectors to avoid mixing different chunking strategies
  console.log("🧹 Clearing existing vectors from Pinecone...");
  await index.deleteAll();

  const vectors = await Promise.all(
    chunks.map(async (chunk, i) => {
      const embedding = await getEmbedding(chunk);
      return {
        id: `chunk-${i}`,
        values: embedding,
        metadata: { text: chunk },
      };
    })
  );

  await index.upsert({ records: vectors as any });
  console.log(`✅ Stored ${vectors.length} chunks in Pinecone`);
}
