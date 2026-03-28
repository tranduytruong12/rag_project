"""
Evaluation — Abstract base evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """
    Abstract evaluator for RAG quality metrics.

    Each concrete evaluator measures one aspect of RAG quality
    (faithfulness, answer relevance, context precision, …).
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Human-readable name of this metric."""
        ...

    @abstractmethod
    def evaluate(self, sample: dict[str, Any]) -> float:
        """
        Evaluate a single sample and return a scalar score in [0, 1].

        Args:
            sample: A dict with keys like:
                - "question"    (str)
                - "answer"      (str)  — generated answer
                - "contexts"    (list[str]) — retrieved chunk texts
                - "ground_truth" (str) — reference answer (if available)

        Returns:
            Float score in [0.0, 1.0] (higher = better).
        """
        ...

    def evaluate_dataset(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        """
        Evaluate a list of samples and return aggregate statistics.

        Returns:
            Dict with "mean", "min", "max" scores for this metric.
        """
        scores = [self.evaluate(s) for s in samples]
        if not scores:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
        }
