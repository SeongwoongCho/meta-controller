# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite import point_mass
from dm_control.utils import io as resources
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'point_mass_joint.xml')), common.ASSETS

def get_model_and_assets_tendon(tendon_coef):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, f'point_mass_tendon{tendon_coef}.xml')), common.ASSETS

@point_mass.SUITE.add('custom')
def joint_easy(time_limit=point_mass._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = point_mass.Physics.from_xml_string(*get_model_and_assets())
  task = point_mass.PointMass(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@point_mass.SUITE.add('custom')
def tendon_nine_easy(time_limit=point_mass._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = point_mass.Physics.from_xml_string(*get_model_and_assets_tendon(0.9))
  task = point_mass.PointMass(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@point_mass.SUITE.add('custom')
def tendon_five_easy(time_limit=point_mass._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = point_mass.Physics.from_xml_string(*get_model_and_assets_tendon(0.5))
  task = point_mass.PointMass(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)
