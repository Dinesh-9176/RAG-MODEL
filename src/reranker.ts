/**
 * LLM-as-a-Reranker Module
 *
 * Uses the local Ollama LLM to re-score retrieved chunks by semantic relevance
 * to the original query. This is a "cross-encoder style" reranking approach
 * where the LLM sees both the query and each candidate passage together,
 * enabling deeper relevance understanding than vector similarity alone.
 *
 * Why LLM-based reranking?
 *   - No external API keys needed (fully local, like the rest of the pipeline)
 *   - Cross-attention between query and passage captures nuance that
 *     bi-encoder (embedding) similarity misses
 *   - Works with any Ollama model already pulled
 */

const OLLAMA_URL = "http://localhost:11434/api/generate";
const RERANKER_MODEL = "llama3.2:3b";

/** Shape of a chunk after reranking */
export interface RankedChunk {
  text: string;
  originalScore: number;   // Pinecone cosine similarity
  rerankerScore: number;   // LLM relevance score (0–10)
  originalIndex: number;   // Position in the original retrieval list
}

/**
 * Ask the LLM to rate how relevant a passage is to a query (0–10).
 * Uses low temperature and tight token budget for speed & consistency.
 */
async function scoreChunk(query: string, chunk: string): Promise<number> {
  // Truncate very long chunks to keep the prompt manageable
  const truncatedChunk = chunk.length > 1500 ? chunk.substring(0, 1500) + "..." : chunk;

  const response = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: RERANKER_MODEL,
      prompt: `You are a Relevance Judge. Rate how relevant the following passage is to the query.

Query: "${query}"

Passage: "${truncatedChunk}"

Scoring rules:
- 0 = completely irrelevant, no connection to the query
- 1-3 = marginally relevant, mentions related topics but doesn't answer the query
- 4-6 = moderately relevant, contains some useful information for the query
- 7-9 = highly relevant, directly addresses the query with specific details
- 10 = perfectly relevant, fully and precisely answers the query

Return ONLY a single integer between 0 and 10. No explanation.

Relevance Score:`,
      stream: false,
      options: {
        num_predict: 5,    // Only need a single number
        temperature: 0,    // Deterministic scoring
      },
    }),
  });

  if (!response.ok) return 5; // Neutral fallback if LLM call fails

  const data: any = await response.json();
  const raw = data.response.trim();

  // Extract the first number found in the response
  const match = raw.match(/\d+/);
  if (!match) return 5;

  const score = parseInt(match[0], 10);
  return Math.max(0, Math.min(10, score)); // Clamp to 0–10
}

/**
 * Rerank a list of retrieved chunks using the local LLM.
 *
 * Flow:
 *   1. Score every candidate chunk in parallel for speed
 *   2. Sort by reranker score (LLM relevance), using original vector
 *      similarity as a tiebreaker
 *   3. Return the top-N most relevant chunks
 *
 * @param query     - The user's original question (not the rewritten one)
 * @param chunks    - Candidate chunks from Pinecone with their similarity scores
 * @param topN      - How many chunks to keep after reranking (default: 5)
 * @returns         - The top-N chunks sorted by LLM-judged relevance
 */
export async function rerankChunks(
  query: string,
  chunks: { text: string; score: number }[],
  topN: number = 5
): Promise<RankedChunk[]> {
  console.log(`🔀 Reranking ${chunks.length} chunks using LLM-as-a-Judge...`);

  // Score all chunks in parallel for maximum speed
  const scored = await Promise.all(
    chunks.map(async (chunk, index) => {
      const rerankerScore = await scoreChunk(query, chunk.text);
      return {
        text: chunk.text,
        originalScore: chunk.score,
        rerankerScore,
        originalIndex: index,
      };
    })
  );

  // Sort by reranker score (descending), then by original Pinecone score as tiebreaker
  scored.sort((a, b) => {
    if (b.rerankerScore !== a.rerankerScore) return b.rerankerScore - a.rerankerScore;
    return b.originalScore - a.originalScore;
  });

  const topChunks = scored.slice(0, topN);

  // Log reranking results
  console.log(`\n📊 Reranking Results (kept top ${topN} of ${chunks.length}):`);
  scored.forEach((chunk, i) => {
    const kept = i < topN ? "✅" : "❌";
    console.log(
      `  ${kept} [Original #${chunk.originalIndex + 1}] ` +
      `Vector: ${chunk.originalScore.toFixed(4)} → ` +
      `LLM Rerank: ${chunk.rerankerScore}/10 ` +
      `| ${chunk.text.substring(0, 80)}...`
    );
  });
  console.log("");

  return topChunks;
}
