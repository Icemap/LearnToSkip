"""Experiment registry."""
from learn_to_skip.experiments.motivation import MotivationExperiment
from learn_to_skip.experiments.build_speed import BuildSpeedExperiment
from learn_to_skip.experiments.recall import RecallExperiment
from learn_to_skip.experiments.classifier_analysis import ClassifierAnalysisExperiment
from learn_to_skip.experiments.ablation import AblationExperiment
from learn_to_skip.experiments.threshold import ThresholdSensitivityExperiment
from learn_to_skip.experiments.scalability import ScalabilityExperiment
from learn_to_skip.experiments.generalization import GeneralizationExperiment
from learn_to_skip.experiments.adaptive_ef import AdaptiveEfExperiment

EXPERIMENT_REGISTRY = {
    "motivation": MotivationExperiment,
    "build_speed": BuildSpeedExperiment,
    "recall": RecallExperiment,
    "classifier": ClassifierAnalysisExperiment,
    "ablation": AblationExperiment,
    "threshold": ThresholdSensitivityExperiment,
    "scalability": ScalabilityExperiment,
    "generalization": GeneralizationExperiment,
    "adaptive": AdaptiveEfExperiment,
}
