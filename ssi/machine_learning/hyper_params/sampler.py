from sklearn.model_selection import ParameterSampler
from .hyper_params import ParamDistributions, FeatureExtractorType


def create_sampler_for_model(model_type: str, number_of_iterations: int, random_state: int = 42) -> ParameterSampler:
    param_distributions = ParamDistributions.distributions_for_model(
        model_type)
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)


def create_sampler_for_pipeline(feature_extractor: FeatureExtractorType, model_type: str, number_of_iterations: int, random_state: int = 42) -> ParameterSampler:
    param_distributions = ParamDistributions.distributions_for_pipeline(
        feature_extractor, model_type)
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)
