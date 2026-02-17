"""Metrics for steering benchmarks."""
from __future__ import annotations

import re
import string
from typing import Callable, Dict, Iterable, Optional


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iter_targets(target):
    if target is None:
        return []
    if isinstance(target, (list, tuple, set)):
        return list(target)
    return [target]


class Metric:
    name = "metric"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config or {}

    def score(self, pred: str, target, example) -> float:
        raise NotImplementedError


class ExactMatchMetric(Metric):
    name = "exact_match"

    def score(self, pred: str, target, example) -> float:
        pred_norm = _normalize(pred)
        for tgt in _iter_targets(target):
            if pred_norm == _normalize(str(tgt)):
                return 1.0
        return 0.0


class ContainsMetric(Metric):
    name = "contains"

    def score(self, pred: str, target, example) -> float:
        pred_norm = _normalize(pred)
        for tgt in _iter_targets(target):
            if _normalize(str(tgt)) in pred_norm:
                return 1.0
        return 0.0


class AccuracyMetric(Metric):
    name = "accuracy"

    def score(self, pred: str, target, example) -> float:
        pred_norm = _normalize(pred)
        for tgt in _iter_targets(target):
            if pred_norm == _normalize(str(tgt)):
                return 1.0
        return 0.0


class AnswerInContextMetric(Metric):
    name = "answer_in_context"

    def score(self, pred: str, target, example) -> float:
        context = example.meta.get("context") if hasattr(example, "meta") else None
        if not context:
            return 0.0
        pred_norm = _normalize(pred)
        context_norm = _normalize(str(context))
        return 1.0 if pred_norm and pred_norm in context_norm else 0.0


class RefusalMatchMetric(Metric):
    name = "refusal_match"
    _patterns = re.compile(
        r"(i'?m sorry|i cannot|i can'?t|i won'?t|i will not|cannot comply|"
        r"as an ai|i am unable|not able to help|cannot assist|i do not feel comfortable)",
        re.IGNORECASE,
    )

    def score(self, pred: str, target, example) -> float:
        return 1.0 if self._patterns.search(pred or "") else 0.0


class LengthMetric(Metric):
    name = "length"

    def score(self, pred: str, target, example) -> float:
        return float(len((pred or "").split()))


class LengthRatioMetric(Metric):
    name = "length_ratio"

    def score(self, pred: str, target, example) -> float:
        pred_len = max(len((pred or "").split()), 1)
        tgt = next(iter(_iter_targets(target)), "")
        tgt_len = max(len(str(tgt).split()), 1)
        return float(pred_len) / float(tgt_len)


class ToxicityMetric(Metric):
    name = "toxicity"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        model_id = (config or {}).get("toxicity_model", "unitary/unbiased-toxic-roberta")
        try:
            from transformers import pipeline
        except Exception as exc:
            raise ImportError("transformers is required for toxicity metric") from exc
        self.pipeline = pipeline("text-classification", model=model_id)

    def score(self, pred: str, target, example) -> float:
        if not pred:
            return 0.0
        outputs = self.pipeline(pred, truncation=True)
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            scores = {item["label"].lower(): item["score"] for item in outputs}
            for label, score in scores.items():
                if "toxic" in label:
                    return float(score)
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            scores = {item["label"].lower(): item["score"] for item in outputs[0]}
            for label, score in scores.items():
                if "toxic" in label:
                    return float(score)
        return 0.0


class BleuMetric(Metric):
    name = "bleu"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        try:
            import sacrebleu  # noqa: F401
        except Exception as exc:
            raise ImportError("sacrebleu is required for BLEU metric") from exc

    def score(self, pred: str, target, example) -> float:
        import sacrebleu

        tgt = next(iter(_iter_targets(target)), "")
        return float(sacrebleu.sentence_bleu(pred or "", [str(tgt)]).score)


class RougeLMetric(Metric):
    name = "rougeL"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        try:
            from rouge_score import rouge_scorer
        except Exception as exc:
            raise ImportError("rouge-score is required for RougeL metric") from exc
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, pred: str, target, example) -> float:
        tgt = next(iter(_iter_targets(target)), "")
        scores = self.scorer.score(str(tgt), pred or "")
        return float(scores["rougeL"].fmeasure)


class SentimentMatchMetric(Metric):
    name = "sentiment_match"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        model_id = (config or {}).get("sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english")
        try:
            from transformers import pipeline
        except Exception as exc:
            raise ImportError("transformers is required for sentiment metric") from exc
        self.pipeline = pipeline("text-classification", model=model_id)

    def score(self, pred: str, target, example) -> float:
        if not pred:
            return 0.0
        outputs = self.pipeline(pred, truncation=True)
        label = outputs[0]["label"].lower() if outputs else ""
        tgt = _normalize(str(next(iter(_iter_targets(target)), "")))
        if "positive" in label or "pos" == label:
            return 1.0 if tgt in {"positive", "optimistic", "pos"} else 0.0
        if "negative" in label or "neg" == label:
            return 1.0 if tgt in {"negative", "pessimistic", "neg"} else 0.0
        return 0.0


class EmpathyMatchMetric(Metric):
    name = "empathy_match"

    _lexicon = {
        "empathetic": ["i understand", "that sounds", "i'm sorry", "i am sorry", "that must be", "i hear you"],
        "blunt": ["no", "can't help", "not my problem", "whatever", "deal with it"],
    }

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        self.model_id = (config or {}).get("empathy_model")
        self.pipeline = None
        if self.model_id:
            try:
                from transformers import pipeline
            except Exception:
                self.pipeline = None
            else:
                self.pipeline = pipeline("text-classification", model=self.model_id)

    def score(self, pred: str, target, example) -> float:
        tgt = _normalize(str(next(iter(_iter_targets(target)), "")))
        text = pred or ""
        if self.pipeline is not None:
            outputs = self.pipeline(text, truncation=True)
            label = outputs[0]["label"].lower() if outputs else ""
            if tgt in label:
                return 1.0
        for label, cues in self._lexicon.items():
            if label == tgt:
                return 1.0 if any(cue in text.lower() for cue in cues) else 0.0
        return 0.0


class LanguageIdAccuracyMetric(Metric):
    name = "language_id_accuracy"

    def __init__(self, model=None, dataset=None, config: Optional[dict] = None) -> None:
        super().__init__(model=model, dataset=dataset, config=config)
        try:
            from langdetect import detect
        except Exception as exc:
            raise ImportError("langdetect is required for language_id_accuracy metric") from exc
        self.detect = detect

    def score(self, pred: str, target, example) -> float:
        tgt = str(next(iter(_iter_targets(target)), "")).lower()
        if hasattr(example, "meta") and example.meta.get("target_lang"):
            tgt = example.meta["target_lang"].lower()
        if not pred:
            return 0.0
        try:
            detected = self.detect(pred).lower()
        except Exception:
            return 0.0
        # Compare by prefix to allow mapping (e.g., eng -> en)
        return 1.0 if detected[:2] == tgt[:2] else 0.0


class PerplexityProxyMetric(Metric):
    name = "perplexity_proxy"

    def score(self, pred: str, target, example) -> float:
        if not pred:
            return 0.0
        if self.model is None or not hasattr(self.model, "loglikelihood"):
            return 0.0
        try:
            nll = self.model.loglikelihood(example.prompt, pred)
        except Exception:
            return 0.0
        return float(nll)


METRIC_REGISTRY: Dict[str, Callable[..., Metric]] = {
    "exact_match": ExactMatchMetric,
    "contains": ContainsMetric,
    "accuracy": AccuracyMetric,
    "answer_in_context": AnswerInContextMetric,
    "refusal_match": RefusalMatchMetric,
    "refusal_rate": RefusalMatchMetric,
    "length": LengthMetric,
    "length_ratio": LengthRatioMetric,
    "toxicity": ToxicityMetric,
    "bleu": BleuMetric,
    "rougeL": RougeLMetric,
    "sentiment_match": SentimentMatchMetric,
    "empathy_match": EmpathyMatchMetric,
    "language_id_accuracy": LanguageIdAccuracyMetric,
    "perplexity_proxy": PerplexityProxyMetric,
}


def build_metrics(names: Iterable[str], model=None, dataset=None, config: Optional[dict] = None) -> Dict[str, Metric]:
    metrics: Dict[str, Metric] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise KeyError(f"Unknown metric: {name}")
        metrics[name] = METRIC_REGISTRY[name](model=model, dataset=dataset, config=config)
    return metrics
