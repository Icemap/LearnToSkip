"""Classifier registry."""
from learn_to_skip.classifiers.base import BaseSkipClassifier, ClassifierMetrics
from learn_to_skip.classifiers.logistic import LogisticRegressionClassifier
from learn_to_skip.classifiers.svm import LinearSVMClassifier
from learn_to_skip.classifiers.tree import DecisionTreeClassifier
from learn_to_skip.classifiers.xgboost_clf import XGBoostClassifier
from learn_to_skip.classifiers.threshold import ThresholdStrategy, ThresholdSweep

CLASSIFIER_REGISTRY: dict[str, type[BaseSkipClassifier]] = {
    "logistic": LogisticRegressionClassifier,
    "svm": LinearSVMClassifier,
    "tree": DecisionTreeClassifier,
    "xgboost": XGBoostClassifier,
}


def get_classifier(name: str) -> BaseSkipClassifier:
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIER_REGISTRY)}")
    return CLASSIFIER_REGISTRY[name]()


__all__ = [
    "BaseSkipClassifier", "ClassifierMetrics", "get_classifier", "CLASSIFIER_REGISTRY",
    "LogisticRegressionClassifier", "LinearSVMClassifier",
    "DecisionTreeClassifier", "XGBoostClassifier",
    "ThresholdStrategy", "ThresholdSweep",
]
