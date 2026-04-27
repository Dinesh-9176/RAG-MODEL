/**
 * RAG Pipeline — Main Entry Point
 *
 * Full Flow:
 *   document.txt
 *        ↓
 *   structuralChunk() → Ollama nomic-embed-text (local embeddings)
 *        ↓
 *   storeChunks() → Pinecone (stores vectors)
 *        ↓
 *   queryRAG("your question")
 *        ↓
 *   Query Rewrite → nomic-embed-text → search Pinecone → top 20 candidates
 *        ↓
 *   LLM Reranker (llama3.2:3b) → re-scores & keeps top 5 chunks
 *        ↓
 *   Gemma (local) → generates answer 🎉
 */

import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";
const mammoth = require("mammoth");
import { semanticChunk, structuralChunk } from "./chunker";
import { storeChunks } from "./pinecone";
import { queryRAG } from "./retriever";

dotenv.config();

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command) {
    console.log(`
🤖 RAG Pipeline — Ollama + Pinecone

Usage:
  npx ts-node src/index.ts ingest <file>    Chunk & store a document
  npx ts-node src/index.ts query  "<question>"  Ask a question

Examples:
  npx ts-node src/index.ts ingest document.txt
  npx ts-node src/index.ts query "What is the main topic?"
    `);
    return;
  }

  if (command === "ingest") {
    // ── Ingest Flow ──────────────────────────────────────────────
    const filePath = args.slice(1).join(" ");
    if (!filePath) {
      console.error("❌ Please provide a file path: ingest <file>");
      process.exit(1);
    }

    const resolvedPath = path.resolve(filePath);
    if (!fs.existsSync(resolvedPath)) {
      console.error(`❌ File not found: ${resolvedPath}`);
      process.exit(1);
    }

    let text = "";
    if (resolvedPath.toLowerCase().endsWith(".docx")) {
      console.log(`📄 Extracting text from DOCX: ${resolvedPath}`);
      const result = await mammoth.extractRawText({ path: resolvedPath });
      text = result.value;
    } else {
      text = fs.readFileSync(resolvedPath, "utf-8");
    }

    console.log(`📄 Loaded document (${text.length} chars)`);

    // 1. Structural chunking based on headers
    const chunks = structuralChunk(text);

    console.log("\n📦 Chunks:");
    chunks.forEach((chunk, i) => {
      console.log(`  [${i}] ${chunk.substring(0, 80)}...`);
    });

    // 2. Store in Pinecone
    await storeChunks(chunks);
    console.log("\n🎉 Ingestion complete!");
  } else if (command === "query") {
    // ── Query Flow ───────────────────────────────────────────────
    const question = args[1];
    if (!question) {
      console.error('❌ Please provide a question: query "<your question>"');
      process.exit(1);
    }

    console.log(`\n❓ Question: ${question}\n`);
    const answer = await queryRAG(question);
    console.log(`\n💡 Answer:\n${answer}`);
  } else {
    console.error(`❌ Unknown command: "${command}". Use "ingest" or "query".`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("❌ Fatal error:", err);
  process.exit(1);
});
