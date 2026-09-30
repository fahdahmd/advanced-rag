import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from src.config import ROOT, settings
from src.corpus import normalize
from src.llm import chat

EVAL_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

JUDGE_PROMPT = """You are grading an answer to a question about an SEC filing.

Question: {question}
Correct answer: {gold}
Model answer: {answer}

Is the model answer correct? It must state the same key fact or number as the correct answer.
Rounding and unit formatting are fine ($416.2 billion = $416,161 million). Extra correct detail is fine.
A wrong number, wrong company or year, or NOT_FOUND is incorrect.
Reply with exactly one word: CORRECT or INCORRECT."""


def get_pipelines() -> dict:
    from src import rag_baseline

    return {"baseline": rag_baseline.answer}  # later milestones add entries here


def evidence_recall(evidence: list[dict], contexts: list[str]) -> float:
    """Share of evidence quotes that appear in the retrieved chunks."""
    chunks = [normalize(c) for c in contexts]
    found = sum(any(normalize(e["quote"]) in c for c in chunks) for e in evidence)
    return found / len(evidence)


def judge(q: dict, answer: str) -> bool:
    if q["category"] == "unanswerable":
        return "NOT_FOUND" in answer.upper()
    prompt = JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], answer=answer)
    verdict = chat([{"role": "user", "content": prompt}]).upper()
    return "CORRECT" in verdict and "INCORRECT" not in verdict


def pct(values: list) -> float | None:
    return round(100 * sum(values) / len(values), 1) if values else None


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["category"] != "unanswerable"]
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["correct"])
    return {
        "questions": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "accuracy_pct": pct([r["correct"] for r in results]),
        "accuracy_by_category_pct": {c: pct(v) for c, v in sorted(by_category.items())},
        "recall_pct": pct([r["recall"] for r in answerable if r["recall"] is not None]),
        "false_refusal_pct": pct([r["refused"] for r in answerable]),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 1) if results else None,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pipelines = get_pipelines()
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=list(pipelines))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()

    lines = EVAL_PATH.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    if args.split != "all":
        questions = [q for q in questions if q["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Running '{args.pipeline}' on {len(questions)} '{args.split}' questions with {settings.llm_model}")

    pipeline = pipelines[args.pipeline]
    results = []
    for q in tqdm(questions):
        rec = {
            "id": q["id"], "category": q["category"], "question": q["question"], "gold": q["answer"],
            "answer": "", "contexts": [], "correct": False, "refused": False,
            "recall": None, "latency_s": 0.0, "error": None,
        }
        start = time.time()
        try:
            out = pipeline(q["question"])
            rec["latency_s"] = round(time.time() - start, 1)  # pipeline time only, judge excluded
            rec["answer"], rec["contexts"] = out["answer"], out["contexts"]
            rec["refused"] = "NOT_FOUND" in out["answer"].upper()
            if q["evidence"]:
                rec["recall"] = evidence_recall(q["evidence"], out["contexts"])
            rec["correct"] = judge(q, out["answer"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)

    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        first = next(r for r in results if r["error"])
        print(f"\nWARNING: {summary['errors']} question(s) hit errors. First error:\n  {first['error'][:400]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{args.pipeline}_{args.split}_{stamp}.json"
    payload = {"pipeline": args.pipeline, "split": args.split, "model": settings.llm_model,
               "timestamp": stamp, "summary": summary, "results": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()