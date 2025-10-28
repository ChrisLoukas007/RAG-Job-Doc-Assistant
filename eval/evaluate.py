"""
1. Reads test questions and their correct answers from a file
2. Asks my AI chatbot each question and gets its response
3. Compares the AI's answers with the correct ones using math
4. Gives my AI a score showing how well it performed
5. Saves all results so I can review them later
6. Logs everything to MLflow for experiment tracking
"""

from __future__ import annotations  # Makes Python understand newer type hints

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
from httpx import HTTPError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# SETUP: Import MLflow if available (MLflow = experiment tracking tool)
try:
    import mlflow  # This tool helps track my AI experiments

    USE_MLFLOW = True
    print("✅ MLflow found - will track experiments")
except Exception:
    mlflow = None
    USE_MLFLOW = False
    print("⚠️ MLflow not installed - will skip experiment tracking")

# SETUP: Load the AI model that converts text to numbers (embeddings)
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

EMB = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


# HELPER FUNCTIONS
def cos(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate how similar two sentences are (0 = completely different, 1 = identical)
    """
    return float(cosine_similarity(a, b)[0, 0])


async def ask(q: str, url: str) -> Dict[str, Any]:
    """
    Send a question to my AI chatbot and measure how long it takes to respond

    Args:
        q: The question to ask (like "What is Python?")
        url: Where my chatbot lives (like http://localhost:8000/query)

    Returns:
        Dictionary with the answer, response time, and raw data
    """
    print(f"❓ Asking: {q[:50]}...")  # Show first 50 characters of question

    async with httpx.AsyncClient(timeout=60.0) as c:
        # Start timing
        t0 = time.perf_counter()

        # Send question to my chatbot
        r = await c.post(url, json={"question": q})

        # Stop timing
        latency_s = time.perf_counter() - t0

        # Make sure we got a good response
        r.raise_for_status()
        data = r.json()

        print(f"Got answer in {latency_s * 1000:.1f}ms")
        return {"answer": data.get("answer", ""), "latency_s": latency_s, "raw": data}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read test questions and answers from a file
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    questions_and_answers = [json.loads(line) for line in lines if line.strip()]
    print(f"Loaded {len(questions_and_answers)} test questions")
    return questions_and_answers


# 🎯 MAIN EVALUATION LOGIC
async def main():
    print("🚀 Starting RAG Evaluation...")
    print("=" * 60)

    # STEP 1: Set up file paths and URLs (configurable via environment)
    query_url = os.getenv("EVAL_QUERY_URL", "http://localhost:8000/query")
    ds_path = Path(os.getenv("EVAL_DATASET_PATH", "eval/questions.jsonl"))
    preds_out = Path(os.getenv("EVAL_PREDICTIONS_PATH", "eval/predictions.jsonl"))

    print(f"Will ask questions to: {query_url}")
    print(f"Reading test data from: {ds_path}")
    print(f"Will save results to: {preds_out}")
    print()

    # STEP 2: Load my test dataset (questions with correct answers)
    gold = read_jsonl(ds_path)  # "gold" = correct answers
    print()

    # STEP 3: Ask my AI chatbot each question (one by one for safety)
    print("Starting to ask questions to my AI...")
    preds: List[Dict[str, Any]] = []  # "preds" = predictions (AI's answers)

    for i, ex in enumerate(gold, 1):
        print(f"Question {i}/{len(gold)}")
        try:
            # Ask the question and get response
            resp = await ask(ex["question"], url=query_url)
        except HTTPError as e:
            # If the server crashes on a question, keep going with other questions
            print(f"Error asking question: {e}")
            resp = {"answer": "", "latency_s": 0.0, "raw": {"error": str(e)}}
        preds.append(resp)

    print()
    print("Finished asking all questions!")
    print()

    # STEP 4: Compare AI answers with correct answers using math
    print("Comparing AI answers with correct answers...")

    sims: List[float] = []  # Similarity scores (0-1)
    results: List[Dict[str, Any]] = []  # Detailed results for each question

    for ex, pred in zip(gold, preds):
        # Convert both answers to numbers (embeddings) so we can compare them
        ai_answer_numbers = EMB.encode([pred["answer"]])
        correct_answer_numbers = EMB.encode([ex["answer"]])

        # Calculate similarity (0 = completely different, 1 = identical)
        similarity_score = cos(ai_answer_numbers, correct_answer_numbers)
        sims.append(similarity_score)

        # Save detailed info for this question
        results.append(
            {
                "question": ex["question"],
                "correct_answer": ex["answer"],
                "ai_answer": pred["answer"],
                "similarity_score": similarity_score,
                "response_time_ms": round(pred["latency_s"] * 1000, 1),
            }
        )

    # STEP 5: Calculate overall performance metrics
    mean_sim = float(np.mean(sims)) if sims else 0.0  # Average similarity
    median_sim = float(np.median(sims)) if sims else 0.0  # Middle similarity
    pct_above_0_7 = (
        float(np.mean([s >= 0.7 for s in sims])) if sims else 0.0
    )  # % good answers
    avg_latency_ms = (
        float(np.mean([r["response_time_ms"] for r in results])) if results else 0.0
    )

    # STEP 6: Show results to the user
    print("EVALUATION RESULTS")
    print("=" * 30)
    print(f"Average similarity:     {mean_sim:.3f} (higher = better, max = 1.0)")
    print(f"Median similarity:      {median_sim:.3f}")
    print(f"Good answers (≥70%):    {pct_above_0_7 * 100:.1f}%")
    print(f"Average response time:  {avg_latency_ms:.1f}ms")
    print()

    # Interpret the results for beginners
    if mean_sim >= 0.8:
        print("EXCELLENT! my AI is performing very well!")
    elif mean_sim >= 0.7:
        print("GOOD! my AI is performing well, with room for improvement.")
    elif mean_sim >= 0.6:
        print("OKAY. my AI needs some tuning to improve accuracy.")
    else:
        print("NEEDS WORK. my AI needs significant improvements.")

    # STEP 7: Save detailed results to a file for later review
    preds_out.parent.mkdir(parents=True, exist_ok=True)  # Create folders if needed
    preds_out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    print(f"Detailed results saved to: {preds_out}")
    print()

    # STEP 8:Log everything to MLflow for experiment tracking
    if USE_MLFLOW and mlflow is not None:
        print("Logging to MLflow for experiment tracking...")

        # MLflow helps you track different versions of my AI and compare them
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "rag-eval"))
        run_name = os.getenv("RUN_NAME", "baseline")

        with mlflow.start_run(run_name=run_name):
            # Log configuration (what settings you used for this experiment)
            mlflow.log_params(
                {
                    "embedding_model": DEFAULT_EMBEDDING_MODEL,
                    "retriever_top_k": os.getenv("RETRIEVER_TOP_K", "default"),
                    "chunk_size": os.getenv("CHUNK_SIZE", "default"),
                    "chunk_overlap": os.getenv("CHUNK_OVERLAP", "default"),
                    "llm_provider": os.getenv("LLM_PROVIDER", "my-llm"),
                    "llm_model": os.getenv("LLM_MODEL", "model-name"),
                    "temperature": os.getenv("TEMPERATURE", "default"),
                }
            )

            # Log performance metrics (how well my AI performed)
            mlflow.log_metric("average_similarity", float(mean_sim))
            mlflow.log_metric("median_similarity", float(median_sim))
            mlflow.log_metric("percent_good_answers", float(pct_above_0_7 * 100))
            mlflow.log_metric("num_test_questions", int(len(gold)))
            mlflow.log_metric("avg_response_time_ms", float(avg_latency_ms))

            # Log metadata (extra info about this experiment)
            try:
                ds_hash = hashlib.md5(ds_path.read_bytes()).hexdigest()
            except FileNotFoundError:
                ds_hash = "file_missing"

            mlflow.set_tag("dataset_path", str(ds_path))
            mlflow.set_tag("dataset_checksum", ds_hash)
            mlflow.set_tag("query_url", query_url)

            # Save files to MLflow (so you can download them later)
            if ds_path.exists():
                mlflow.log_artifact(str(ds_path), artifact_path="test_data")
            mlflow.log_artifact(str(preds_out), artifact_path="results")

        print("Results logged to MLflow!")
        print("View results at: http://localhost:5000 (run 'mlflow ui' to start)")
    else:
        print("MLflow not available - skipping experiment tracking")

    print()
    print("Evaluation complete!")
    print("=" * 60)


#    RUN THE SCRIPT
if __name__ == "__main__":
    # This runs the evaluation when you execute this file
    asyncio.run(main())
