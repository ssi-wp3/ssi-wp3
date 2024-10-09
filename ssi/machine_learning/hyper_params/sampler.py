from sklearn.model_selection import ParameterSampler
from .hyper_params import ParamDistributions, FeatureExtractorType, ModelType


def create_sampler_for_model(model_type: ModelType,
                             number_of_iterations: int,
                             random_state: int = 42) -> ParameterSampler:
    """ Create a parameter sampler for the model.

    Parameters
    ----------
    model_type : ModelType
        The type of model.
    number_of_iterations : int
        The number of iterations for the sampler.
    random_state : int, optional
        The random state for the sampler, by default 42.

    Returns
    -------
    sklearn.model_selection.ParameterSampler
        The sampler for the model.
    """
    param_distributions = ParamDistributions.distributions_for_model(
        model_type)
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)


def create_sampler_for_pipeline(feature_extractor: FeatureExtractorType,
                                model_type: ModelType,
                                number_of_iterations: int,
                                feature_extractor_prefix: str,
                                model_prefix: str,
                                random_state: int = 42) -> ParameterSampler:
    """ Create a parameter sampler for the pipeline.

    Parameters
    ----------
    feature_extractor : FeatureExtractorType
        The type of feature extractor.
    model_type : ModelType
        The type of model.
    number_of_iterations : int
        The number of iterations for the sampler.
    feature_extractor_prefix : str
        The prefix for the feature extractor.
    model_prefix : str
        The prefix for the model.
    random_state : int, optional
        The random state for the sampler, by default 42.

    Returns
    -------
    sklearn.model_selection.ParameterSampler
        The sampler for the pipeline.
    """
    param_distributions = ParamDistributions.distributions_for_pipeline(
        feature_extractor, model_type, f"{feature_extractor_prefix}__", f"{model_prefix}__")
    return ParameterSampler(param_distributions, n_iter=number_of_iterations, random_state=random_state)
