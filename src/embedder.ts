/**
 * Ollama Embedding Module
 * Uses nomic-embed-text model running locally via Ollama
 * to generate 768-dimensional embeddings.
 */

const OLLAMA_BASE_URL = "http://localhost:11434";

/**
 * Get embedding vector for a given text using Ollama's nomic-embed-text model.
 * @param text - The text to embed
 * @param taskType - Whether this is for a 'query' (question) or 'document' (storage)
 * @returns A 768-dimensional embedding vector
 */
export async function getEmbedding(
  text: string,
  taskType: "query" | "document" = "document"
): Promise<number[]> {
  const prefix = taskType === "query" ? "search_query: " : "search_document: ";
  const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "nomic-embed-text",
      prompt: prefix + text,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama embedding request failed: ${response.status} ${response.statusText}`
    );
  }

  const data: any = await response.json();
  return data.embedding;
}
