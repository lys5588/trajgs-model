# coding=utf-8
"""
Reusable text generation evaluator for multi-reference NLG metrics.

Metrics:
- Bleu_1
- Bleu_4
- ROUGE_L
- CIDEr
- METEOR
- Bert_F1
"""

from typing import Dict, List


class TextNLGEvaluator:
    """Compute text-generation metrics with graceful fallback when deps are missing."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.nlg_evaluator = None
        self.has_nlg = False
        self.has_meteor = False
        self._init_nlg_evaluator()

    def _init_nlg_evaluator(self):
        try:
            import os
            import sys

            # Avoid local `datasets` package shadowing HuggingFace `datasets`.
            original_path = sys.path.copy()
            original_modules = {
                k: v
                for k, v in sys.modules.items()
                if k == "datasets" or k.startswith("datasets.")
            }
            keys_to_remove = [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]
            for key in keys_to_remove:
                del sys.modules[key]

            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path = [p for p in sys.path if not p.startswith(project_dir)]

            from nlgmetricverse import NLGMetricverse, load_metric

            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            try:
                metrics.append(load_metric("meteor"))
                self.has_meteor = True
            except Exception as e:
                self.has_meteor = False
                print(f"[TextNLGEvaluator] METEOR metric unavailable, skipping: {e}")
            self.nlg_evaluator = NLGMetricverse(metrics)
            self.has_nlg = True

            # Restore path; no need to restore removed modules.
            sys.path = original_path
            for key, value in original_modules.items():
                if key not in sys.modules:
                    sys.modules[key] = value
        except Exception as e:
            print(f"[TextNLGEvaluator] nlgmetricverse unavailable: {e}")
            print("[TextNLGEvaluator] BLEU/ROUGE/CIDEr/METEOR will be skipped.")
            self.nlg_evaluator = None
            self.has_nlg = False
            self.has_meteor = False

    @staticmethod
    def _normalize_inputs(predictions: List[str], references: List[List[str]]) -> (List[str], List[List[str]]):
        preds = []
        refs = []
        for idx, pred in enumerate(predictions):
            preds.append(pred if isinstance(pred, str) else str(pred))
            cur_refs = references[idx] if idx < len(references) else [""]
            if isinstance(cur_refs, str):
                cur_refs = [cur_refs]
            cur_refs = [r for r in cur_refs if isinstance(r, str) and len(r.strip()) > 0]
            if len(cur_refs) == 0:
                cur_refs = [""]
            refs.append(cur_refs)
        return preds, refs

    def compute(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        if len(predictions) == 0:
            return {}

        preds, refs = self._normalize_inputs(predictions, references)
        metrics: Dict[str, float] = {}

        if self.has_nlg and self.nlg_evaluator is not None:
            try:
                scores = self.nlg_evaluator(predictions=preds, references=refs)
                if "bleu_1" in scores:
                    metrics["Bleu_1"] = float(scores["bleu_1"]["score"])
                if "bleu_4" in scores:
                    metrics["Bleu_4"] = float(scores["bleu_4"]["score"])
                if "rouge" in scores:
                    metrics["ROUGE_L"] = float(scores["rouge"]["rougeL"])
                if "cider" in scores:
                    metrics["CIDEr"] = float(scores["cider"]["score"])
                if "meteor" in scores:
                    meteor_score = scores["meteor"]
                    if isinstance(meteor_score, dict):
                        meteor_score = meteor_score.get("score", meteor_score.get("meteor", 0.0))
                    metrics["METEOR"] = float(meteor_score)
            except Exception as e:
                print(f"[TextNLGEvaluator] NLG metric evaluation failed: {e}")

        try:
            from bert_score import score as score_bert

            _, _, f1 = score_bert(
                preds,
                refs,
                lang="en",
                rescale_with_baseline=True,
                idf=True,
                device=self.device,
                verbose=False,
            )
            metrics["Bert_F1"] = float(f1.mean())
        except Exception as e:
            print(f"[TextNLGEvaluator] BertScore failed: {e}")

        return metrics
