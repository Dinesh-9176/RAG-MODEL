/**
 * RAG Retriever & Generator
 * Queries Pinecone for relevant chunks, reranks them with an LLM judge,
 * and uses Ollama Gemma for answer generation.
 */

import { Pinecone } from "@pinecone-database/pinecone";
import { getEmbedding } from "./embedder";
import { rerankChunks } from "./reranker";
import * as dotenv from "dotenv";
dotenv.config();

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
const index = pc.index("rag-semantic-ts");

/**
 * Generate an answer using Ollama's Gemma model given context and a question.
 */
async function generateAnswer(
  context: string,
  question: string
): Promise<string> {
  const response = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llama3.2:3b",
      prompt: `You are a strict evaluation assistant. 
Follow these rules strictly:
1. Use ONLY the provided context below to answer the question.
2. If the answer is not in the context, say: "I am sorry, but the provided documentation does not contain information to answer this question."
3. Do not use any outside knowledge or provide "common sense" explanations.
4. Be concise and direct.

Context:
${context}

Question: ${question}

Answer (based strictly on context):`,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama generate request failed: ${response.status} ${response.statusText}`
    );
  }

  const data: any = await response.json();
  return data.response;
}

import { calculateFaithfulness, calculateRelevance } from "./evaluator";

/**
 * Rewrite the user's question into a more descriptive retrieval query.
 * This helps match specific parameters and scoring rubrics.
 */
async function rewriteQuery(question: string): Promise<string> {
  const response = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llama3.2:3b",
      prompt: `You are a Retrieval Optimizer. Your goal is to rewrite the user's question into a keyword-rich search query for a vector database.
The document contains these 6 parameters: 
1. Relevance of Experience
2. Maturity
3. Drive
4. Problem Solving
5. Learning Orientation
6. Communication

User Question: "${question}"

Rule: Transform shorthand like "Parameter 1" into the full name (e.g., "Parameter 1: Relevance of Experience scoring criteria").
Return ONLY the optimized search query text.

Optimized Search Query:`,
      stream: false,
    }),
  });

  if (!response.ok) return question; // Fallback
  const data: any = await response.json();
  return data.response.trim();
}

/**
 * Full RAG pipeline:
 * 1. Rewrite the question for better retrieval
 * 2. Embed the expanded question
 * 3. Search Pinecone for top-20 candidate chunks
 * 4. Rerank candidates with LLM judge → keep top 5
 * 5. Generate an answer and evaluate
 */
export async function queryRAG(question: string): Promise<string> {
  const startTime = performance.now();

  // 1. Rewrite & Embed the question
  console.log(`🔄 Refining query for better retrieval...`);
  const expandedQuery = await rewriteQuery(question);
  console.log(`🔎 Searching Pinecone for: "${expandedQuery}"`);

  const embedStart = performance.now();
  const queryEmbedding = await getEmbedding(expandedQuery, "query"); // Use expanded for search
  const embedTime = performance.now() - embedStart;

  // 2. Search Pinecone (broad retrieval — cast a wide net)
  const searchStart = performance.now();
  const results = await index.query({
    vector: queryEmbedding,
    topK: 20, // Retrieve 20 candidates for reranking
    includeMetadata: true,
  });
  const searchTime = performance.now() - searchStart;

  console.log(`🔍 Retrieved ${results.matches.length} candidate chunks from Pinecone`);

  // 3. Rerank with LLM — narrow down to the best 5 chunks
  const rerankStart = performance.now();
  const candidateChunks = results.matches.map((match) => ({
    text: (match.metadata?.text as string) || "",
    score: match.score || 0,
  }));
  const rerankedChunks = await rerankChunks(question, candidateChunks, 5);
  const rerankTime = performance.now() - rerankStart;

  // 4. Build context from reranked chunks
  const context = rerankedChunks
    .map((chunk) => chunk.text)
    .join("\n\n");

  const topVectorScore = results.matches[0]?.score || 0;
  const topRerankScore = rerankedChunks[0]?.rerankerScore || 0;

  console.log(`✅ Reranked to top ${rerankedChunks.length} chunks:`);
  rerankedChunks.forEach((chunk, i) => {
    console.log(`--- Reranked Chunk [${i + 1}] (Vector: ${chunk.originalScore.toFixed(4)}, LLM: ${chunk.rerankerScore}/10) ---`);
    console.log(`${chunk.text.substring(0, 200)}...\n`);
  });

  // 5. Generate answer with Gemma
  const genStart = performance.now();
  const answer = await generateAnswer(context, question);
  const genTime = performance.now() - genStart;

  // 6. RAGAS Evaluation
  const evalStart = performance.now();
  console.log(`⚖️  Evaluating RAGAS metrics (Faithfulness & Relevance)...`);
  const [faithfulness, relevance] = await Promise.all([
    calculateFaithfulness(answer, context),
    calculateRelevance(answer, question)
  ]);
  const evalTime = performance.now() - evalStart;

  const totalTime = performance.now() - startTime;

  // Print Performance & Accuracy Metrics
  console.log(`\n📊 PERFORMANCE & RAGAS EVALUATION`);
  console.log(`──────────────────────────────────────`);
  console.log(`🎯 Vector Similarity    : ${(topVectorScore * 100).toFixed(2)}%`);
  console.log(`🔀 Top Rerank Score     : ${topRerankScore}/10`);
  console.log(`✅ Faithfulness (RAGAS) : ${(faithfulness * 100).toFixed(2)}%`);
  console.log(`🎯 Answer Relevance     : ${(relevance * 100).toFixed(2)}%`);
  console.log(`──────────────────────────────────────`);
  console.log(`⏱️  Embedding Latency    : ${embedTime.toFixed(0)}ms`);
  console.log(`⏱️  Search Latency       : ${searchTime.toFixed(0)}ms`);
  console.log(`🔀 Reranking Latency    : ${rerankTime.toFixed(0)}ms`);
  console.log(`⏱️  Generation Latency   : ${genTime.toFixed(0)}ms`);
  console.log(`⚖️  Evaluation Latency   : ${evalTime.toFixed(0)}ms`);
  console.log(`🚀 Total Response Time  : ${totalTime.toFixed(0)}ms`);
  console.log(`──────────────────────────────────────\n`);

  return answer;
}
