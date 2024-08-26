from sklearn.model_selection import ParameterSampler
from .hyper_params import ParamDistributions, FeatureExtractorType, ModelType


def create_sampler_for_model(model_type: ModelType,
                             number_of_iterations: int,
                             random_state: int = 42) -> ParameterSampler:
    param_distributions = ParamDistributions.distributions_for_model(
        model_type)
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)


def create_sampler_for_pipeline(feature_extractor: FeatureExtractorType,
                                model_type: ModelType,
                                number_of_iterations: int,
                                feature_extractor_prefix: str,
                                model_prefix: str,
                                random_state: int = 42) -> ParameterSampler:
    param_distributions = ParamDistributions.distributions_for_pipeline(
        feature_extractor, model_type, f"{feature_extractor_prefix}__", f"{model_prefix}__")
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)
