/**
 * RAGAS-Lite Evaluator
 * Implementation of core RAG evaluation metrics using local LLM-as-a-judge.
 */

import { getEmbedding } from "./embedder";

const OLLAMA_URL = "http://localhost:11434/api/generate";
const JUDGE_MODEL = "llama3.2:3b"; // Switched to faster 3B model

/**
 * Utility to call Ollama for evaluation tasks.
 */
async function callJudge(prompt: string): Promise<string> {
  const response = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: JUDGE_MODEL,
      prompt: prompt + "\n\nResponse (concise):", // Ask for concise output
      stream: false,
      options: {
        num_predict: 100, // Limit lengths to speed up evaluation
      }
    }),
  });

  if (!response.ok) throw new Error("Ollama judge call failed");
  const data: any = await response.json();
  return data.response.trim();
}

/**
 * Calculate Faithfulness Score (Groundedness).
 * 1. Extract facts from answer.
 * 2. Verify facts against context.
 */
export async function calculateFaithfulness(
  answer: string,
  context: string
): Promise<number> {
  // Step 1: Extract facts
  const extractionPrompt = `Extract 3-5 keys facts from this answer as a simple list.
Answer: "${answer}"`;

  const factsRaw = await callJudge(extractionPrompt);
  const facts = factsRaw
    .split("\n")
    .map((f) => f.replace(/^[-*•\d.]\s*/, "").trim())
    .filter((f) => f.length > 5)
    .slice(0, 5); // Limit to top 5 facts to save time

  if (facts.length === 0) return 1.0;

  // Step 2: Verify facts (Parallelized for Speed)
  console.log(`🧪 Verifying ${facts.length} extracted facts:`);
  facts.forEach((f, i) => console.log(`   [${i + 1}] ${f}`));
  
  const verificationResults = await Promise.all(facts.map(async (fact) => {
    const prompt = `### Context:
${context.substring(0, 6000)}

### Fact:
${fact}

### Question:
Is the Fact above supported by the Context provided? Answer only with 'Yes' or 'No'.`;

    const verdict = await callJudge(prompt);
    return verdict.toLowerCase().includes("yes");
  }));

  const supportedCount = verificationResults.filter(v => v).length;
  return supportedCount / facts.length;
}

/**
 * Calculate Answer Relevancy Score.
 * 1. Generate hypothetical question from answer.
 * 2. Compare similarity with original question.
 */
export async function calculateRelevance(
  answer: string,
  originalQuestion: string
): Promise<number> {
  // Step 1: Generate hypothetical question
  const generationPrompt = `Generate a single short direct question that the following text was intended to answer. 
Response should be ONLY the question text.

Answer: "${answer}"
Hypothetical Question:`;

  const hypotheticalQuestion = await callJudge(generationPrompt);

  // Step 2: Compare embeddings
  const originalEmb = await getEmbedding(originalQuestion, "query");
  const hypotheticalEmb = await getEmbedding(hypotheticalQuestion, "query");

  // Calculate cosine similarity
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < originalEmb.length; i++) {
    dot += originalEmb[i] * hypotheticalEmb[i];
    normA += originalEmb[i] * originalEmb[i];
    normB += hypotheticalEmb[i] * hypotheticalEmb[i];
  }
  
  const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return Math.max(0, Math.min(1, similarity)); // Clamp to 0-1
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
