"""
Section Detection Accuracy Report.

Deliverable: "Section detection accuracy report" from the task doc.

Usage:
    python tests/evaluate_section_classifier.py path/to/labeled_samples.json

Labeled sample format (one JSON file, list of resumes):

[
  {
    "id": "resume_001",
    "raw_text": "<full extracted resume text>",
    "ground_truth": [
      {"section": "contact", "text": "Jane Doe | jane@email.com | ..."},
      {"section": "skills", "text": "Python, FastAPI, ..."},
      {"section": "work_experience", "text": "Acme Corp - Engineer ..."},
      ...
    ]
  },
  ...
]

Each ground_truth entry's "text" should be the (trimmed) content of that
section as it appears in raw_text - used to map predicted blocks back to
the correct ground-truth section via substring/best-overlap matching.

Metrics reported:
  - Per-section precision / recall / F1
  - Overall block-level accuracy
  - Breakdown by which tier produced the correct label (rule/nlp/llm)
  - Confusion pairs (most common section A predicted-as section B)

Run with --no-llm to skip the (paid) LLM tier during evaluation and
measure rule+NLP-only accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from utils.hybrid_section_classifier import classify_sections_hybrid
from utils.section_classifier import ResumeSection
from utils.text_cleaner import clean_resume_text


def _best_matching_truth_section(predicted_content: str, ground_truth: list[dict]) -> str:
    """
    Map a predicted block back to whichever ground-truth section it overlaps
    with most (word-overlap heuristic), since block boundaries from the
    classifier won't line up character-for-character with hand-labeled spans.
    """
    pred_words = set(predicted_content.lower().split())
    if not pred_words:
        return "unknown"

    best_section = "unknown"
    best_overlap = 0
    for entry in ground_truth:
        truth_words = set(entry["text"].lower().split())
        overlap = len(pred_words & truth_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_section = entry["section"]

    return best_section if best_overlap > 0 else "unknown"


async def evaluate(samples: list[dict], use_llm: bool) -> dict:
    per_section_tp: Counter = Counter()
    per_section_fp: Counter = Counter()
    per_section_fn: Counter = Counter()
    confidence_correct: Counter = Counter()
    confidence_total: Counter = Counter()
    confusion: Counter = Counter()

    total_blocks = 0
    total_correct = 0

    for sample in samples:
        text = clean_resume_text(sample["raw_text"])
        ground_truth = sample["ground_truth"]
        truth_sections_present = {e["section"] for e in ground_truth}

        predicted_blocks = await classify_sections_hybrid(text, use_llm_fallback=use_llm)

        seen_true_sections = set()

        for block in predicted_blocks:
            predicted = block.section.value
            actual = _best_matching_truth_section(block.content, ground_truth)
            seen_true_sections.add(actual)

            total_blocks += 1
            confidence_total[block.confidence] += 1

            if predicted == actual:
                total_correct += 1
                per_section_tp[actual] += 1
                confidence_correct[block.confidence] += 1
            else:
                per_section_fp[predicted] += 1
                per_section_fn[actual] += 1
                confusion[(actual, predicted)] += 1

        # Any ground-truth section never matched by a predicted block = pure FN
        for missed in truth_sections_present - seen_true_sections:
            per_section_fn[missed] += 1

    report = {"sections": {}, "tiers": {}, "confusion_pairs": {}}

    all_sections = {s.value for s in ResumeSection} | set(per_section_tp) | set(per_section_fn)
    for section in sorted(all_sections):
        tp = per_section_tp[section]
        fp = per_section_fp[section]
        fn = per_section_fn[section]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if tp or fp or fn:
            report["sections"][section] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "support": tp + fn,
            }

    for tier, total in confidence_total.items():
        correct = confidence_correct[tier]
        report["tiers"][tier] = {
            "blocks": total,
            "accuracy": round(correct / total, 3) if total else 0.0,
        }

    report["overall_accuracy"] = round(total_correct / total_blocks, 3) if total_blocks else 0.0
    report["total_blocks"] = total_blocks

    top_confusions = confusion.most_common(10)
    report["confusion_pairs"] = {
        f"{actual} -> predicted {predicted}": count
        for (actual, predicted), count in top_confusions
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate resume section classifier accuracy.")
    parser.add_argument("samples_path", type=str, help="Path to labeled_samples.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM tier (rule+NLP only)")
    parser.add_argument("--out", type=str, default="section_accuracy_report.json")
    args = parser.parse_args()

    samples = json.loads(Path(args.samples_path).read_text())
    report = asyncio.run(evaluate(samples, use_llm=not args.no_llm))

    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}\nSECTION DETECTION ACCURACY REPORT\n{'='*60}")
    print(f"Total samples: {len(samples)}   Total blocks evaluated: {report['total_blocks']}")
    print(f"Overall accuracy: {report['overall_accuracy']*100:.1f}%\n")

    print(f"{'Section':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for section, m in sorted(report["sections"].items()):
        print(f"{section:20s} {m['precision']*100:9.1f}% {m['recall']*100:9.1f}% {m['f1']*100:9.1f}% {m['support']:10d}")

    print(f"\nAccuracy by tier (which stage produced the label):")
    for tier, m in report["tiers"].items():
        print(f"  {tier:20s} {m['blocks']:5d} blocks   {m['accuracy']*100:.1f}% accurate")

    if report["confusion_pairs"]:
        print(f"\nTop confusions (ground truth -> what classifier predicted):")
        for pair, count in report["confusion_pairs"].items():
            print(f"  {pair}: {count}x")

    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()