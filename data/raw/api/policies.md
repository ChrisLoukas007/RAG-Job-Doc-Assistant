# Answering policy

- Use **only** the provided context chunks when answering.
- If the answer is not present, reply: **"I don't know."**
- Always include **citations** (e.g., [S1], [S2]) pointing to the source titles/links.

## No-hit policy
- If no relevant documents are found (under the similarity threshold), return **HTTP 404** with:
  ```json
  { "detail": "No relevant documents were found" }
