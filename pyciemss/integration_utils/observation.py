from pyciemss.observation import NoiseModel, NormalNoiseModel

_STR_TO_OBSERVATION = {"normal": NormalNoiseModel}


def compile_noise_model(model_str: str, **model_kwargs) -> NoiseModel:
    if model_str not in _STR_TO_OBSERVATION.keys():
        raise NotImplementedError(
            f"Noise model {model_str} not implemented. Please select from one of the following: {_STR_TO_OBSERVATION.keys()}"
        )

    return _STR_TO_OBSERVATION[model_str](**model_kwargs)
