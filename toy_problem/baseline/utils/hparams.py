# coding=utf-8
"""Hparams defining 
"""
import six


def merge_hparams(*objects):
  hps = HParams()
  for object_ in objects:
    for k, v in object_.items():
      assert k not in hps._items.keys(), "repeating key {}".format(k)
      hps.set(k, v)
  return hps

class HParams(object):
  """Hparams object to hold hyperparameters
  """
  def __init__(self, **kwargs):
    self._items = {}
    for k, v in kwargs.items():
      self.set(k, v)

  def set(self, k, v):
    self._items[k] = v
    setattr(self, k, v)


  def items(self):
    return self._items.items()

  def __str__(self):
    return str(self._items)

  def parse_dict(self, d):
    """Parses dict, filling inner fields
  
    Args: 
      d: dict object
    """
    hps = HParams(**self._items)
    for key, value in six.iteritems(d):
      assert(key in self._items)
      hps.set(key, value)
    return hps

  def parse(self, str_value):
    """Parses string, filling inner fields
  
    Args: 
      str_value: string
    """
    hps = HParams(**self._items)
    for entry in str_value.strip().split(","):
      entry = entry.strip()
      if not entry:
        continue
      key, sep, value = entry.partition("=")
      if not sep:
        raise ValueError("Unable to parse: %s" % entry)
      default_value = hps._items[key]
      if isinstance(default_value, bool):
        hps.set(key, value.lower() == "true")
      elif isinstance(default_value, int):
        hps.set(key, int(value))
      elif isinstance(default_value, float):
        hps.set(key, float(value))
      else:
        hps.set(key, value)
    return hps
