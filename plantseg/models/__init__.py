from typing import Any, Dict, Type

import torch.nn as nn
from pytorch3dunet.unet3d.model import get_model as _get_unet3d_model

from plantseg.models.models import get_model as _get_model


def get_model(model_config: Dict[str, Any]) -> nn.Module:
  """Loads a model from a config dictionary

  Args:
      model_config (Dict[str, Any]): model config must contain the key 'name'
        which should be one of the keys of _MODEL_CLASSES. The rest of the keys
        should be the arguments of the model class.

  Returns:
      nn.Module: the desired segmentation model.
  """
  assert 'name' in model_config, "Model config should contain a 'name' key"
  try:
    model = _get_unet3d_model(model_config)
  except TypeError:
    model = _get_model(model_config)
  return model
