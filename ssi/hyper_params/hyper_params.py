from typing import Dict, Any
import scipy.stats as stats


LOGISTIC_REGRESSION = {
    "C": stats.loguniform(1e-4, 1e4),
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [100, 200, 300, 400, 500],
    "class_weight": ["balanced", None],
}


DISTRIBUTIONS_FOR_ALL_MODELS = {
    "LogisticRegression": LOGISTIC_REGRESSION,
}


class ParamDistributions:

    @ staticmethod
    def distributions_for_model(model_type: str) -> Dict[str, Any]:
        return DISTRIBUTIONS_FOR_ALL_MODELS[model_type]
