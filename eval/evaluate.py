import json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import httpx, asyncio

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cos(a, b): return float(cosine_similarity(a, b)[0,0])

async def ask(q):
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post("http://localhost:8000/query", json={"question": q})
        r.raise_for_status()
        return r.json()["answer"]

async def main():
    gold = [json.loads(l) for l in open("eval/questions.jsonl")]
    sims = []
    for ex in gold:
        pred = await ask(ex["question"])
        e_pred = EMB.encode([pred])
        e_gold = EMB.encode([ex["answer"]])
        sims.append(cos(e_pred, e_gold))
    print("Mean semantic similarity:", np.mean(sims))

if __name__ == "__main__":
    asyncio.run(main())
