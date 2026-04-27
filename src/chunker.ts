/**
 * Semantic Chunker
 * Splits documents into semantically meaningful chunks using
 * Ollama's nomic-embed-text embeddings to detect topic boundaries.
 */

import { getEmbedding } from "./embedder";

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Split raw text into sentences (basic sentence boundary detection).
 */
function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/**
 * Semantically chunk a document by detecting embedding similarity drops
 * between consecutive sentences, then grouping them into coherent chunks.
 *
 * @param text - The full document text
 * @param similarityThreshold - Below this cosine-similarity, a new chunk starts (default 0.75)
 * @returns An array of text chunks
 */
export async function semanticChunk(
  text: string,
  similarityThreshold = 0.75
): Promise<string[]> {
  const sentences = splitSentences(text);

  if (sentences.length === 0) return [];
  if (sentences.length === 1) return [sentences[0]];

  console.log(`📝 Splitting ${sentences.length} sentences into semantic chunks...`);

  // Embed every sentence using Ollama nomic-embed-text
  const embeddings: number[][] = [];
  for (const sentence of sentences) {
    const emb = await getEmbedding(sentence);
    embeddings.push(emb);
  }

  // Detect chunk boundaries based on cosine similarity drops
  const chunks: string[] = [];
  let currentChunk: string[] = [sentences[0]];

  for (let i = 1; i < sentences.length; i++) {
    const sim = cosineSimilarity(embeddings[i - 1], embeddings[i]);

    if (sim < similarityThreshold) {
      // Similarity drop → start a new chunk
      chunks.push(currentChunk.join(" "));
      currentChunk = [sentences[i]];
    } else {
      currentChunk.push(sentences[i]);
    }
  }

  // Push the last chunk
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(" "));
  }

  console.log(`✅ Created ${chunks.length} semantic chunks`);
  return chunks;
}

/**
 * Structurally chunk a document based on section headers.
 * Looks for patterns like "1. Header" or "Parameter 1:" at the start of lines.
 *
 * @param text - The full document text
 * @returns An array of structural text chunks
 */
export function structuralChunk(text: string): string[] {
  console.log("📂 Performing structural chunking based on headers...");

  // Regex to match ONLY the very top-level Master Headers
  // Capture the entire line to use as a descriptive Section Label
  const headerRegex = /^(?:\d+\.\s+(?:Parameter|How|Hard|The\s+Six)|Parameter\s+\d+:|NOTE\s+FOR\s+AI|OUTLIER(?:S|\s+FLAG)?).*/gm;

  const chunks: string[] = [];
  const matches = [...text.matchAll(headerRegex)];

  if (matches.length === 0) {
    console.warn("⚠️ No structural headers found. Returning entire text as one chunk.");
    return [text];
  }

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i]!.index!;
    const end = matches[i + 1] ? matches[i + 1]!.index! : text.length;
    let chunkContent = text.substring(start, end).trim();

    // SECTION LABELING LOGIC:
    // Extract the title, strip the leading "4. " noise, and prepend a clear label
    const rawHeader = matches[i][0];
    let sectionLabel = rawHeader.replace(/^\d+\.\s+/, "").trim().toUpperCase(); // Strip "4. " -> "PARAMETER 1"
    
    // Add the label to the chunk for the AI to see clearly
    const labeledChunk = `=== [SECTION: ${sectionLabel}] ===\n\n${chunkContent.replace(rawHeader, sectionLabel)}`;

    if (labeledChunk.length > 200) {
      chunks.push(labeledChunk);
    } else {
      console.log(`🧹 Skipping short/TOC chunk: "${sectionLabel}..."`);
    }
  }

  console.log(`✅ Created ${chunks.length} structural chunks`);
  return chunks;
}
