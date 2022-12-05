from typing import Any, Dict, Type

import torch.nn as nn

from plantseg.models.unetr import UNETR

_MODEL_CLASSES: Dict[str, Type[nn.Module]] = {
  'unetr': UNETR
}

def get_model(model_config: Dict[str, Any]) -> nn.Module:
  assert (model_config['name'] in _MODEL_CLASSES,
          f"Model name should be one of {_MODEL_CLASSES.keys()}")
  config = model_config.copy()
  cls = _MODEL_CLASSES[config.pop('name')]
  return cls(**model_config)
