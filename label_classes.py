"""
Label Classes
---------------------------------------------------------
Copyright 2021 YIDING SONG
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

class Label(object):
  def __init__(self, _name, _dtype, _fallback, _key=None, _unit=None):
    self.name = _name
    self.dtype = _dtype
    self.fallback = _fallback
    if _key is None:
      self.key = _name
    else:
      self.key = _key
    self.unit = _unit
  
  def __str__(self):
    return 'Label "{}" of "{}" on key "{}", defaults to {}; unit: {}'.format(
        self.name, self.dtype, self.key, self.fallback, self.unit
    )
  
  def serialize(self):
    return {
      'name': self.name,
      'dtype': self.dtype,
      'fallback': self.fallback,
      'key': self.key,
      'unit': self.unit,
      'labelType': 'Label'
    }

class FloatLabel(Label):
  def __init__(self, _name, **kwargs):
    super().__init__(_name, np.float32, 0, **kwargs)
  
  def fwd_call(self, val):
    return val

  def rev_call(self, val):
    return val
  
  def serialize(self):
    obj = super().serialize()
    obj['labelType'] = 'FloatLabel'
    return obj

class IntClass(Label):
  def __init__(self, _name, _size, **kwargs):
    super().__init__(_name, int, -1, **kwargs)
    self.size = _size
  
  def fwd_call(self, val):
    return round(val)

  def rev_call(self, val):
    return round(val)
  
  def serialize(self):
    obj = super().serialize()
    obj['labelType'] = 'IntClass'
    obj['size'] = self.size
    return obj

class IntClassMap(Label):
  def __init__(self, _name, _map, **kwargs):
    super().__init__(_name, dict, -1, **kwargs)
    self.fwd_map = _map
    self.rev_map = dict(zip(_map.values(), _map.keys()))
   
  def fwd_call(self, val):
    return self.fwd_map[val]
  
  def rev_call(self, val):
    try:
      return self.rev_map[round(val)]
    except:
      return None

  def serialize(self):
    obj = super().serialize()
    obj['labelType'] = 'IntClassMap'
    obj['fwd_map'] = self.fwd_map
    obj['rev_map'] = self.rev_map
    return obj

