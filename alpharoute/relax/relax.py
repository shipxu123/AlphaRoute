"""Analog relaxation."""
from typing import Any, Dict, Sequence, Tuple
from alpharoute.relax import analog_minimize

import torch
import numpy as np

class AnalogRelaxation(object):
  """Analog relaxation."""

  def __init__(self,
               cost_guide_distribution,
               potential_function,
               pool_size: int,
               max_iterations: int,
               max_outer_iterations: int):
    """Initialize Analog Relaxer.

    Args:
      max_iterations: Maximum number of L-BFGS iterations. 0 means no max.
      max_outer_iterations: Maximum number of violation-informed relax
       iterations.
    """
    self._cost_guide_distribution = cost_guide_distribution
    self._potential_function = potential_function
    self._pool_size = pool_size
    self._max_iterations = max_iterations
    self._max_outer_iterations = max_outer_iterations

  def process(self):
    """Runs Analog relax on a prediction, adds noise, returns Cost guide."""
    # out = analog_minimize.run_pipeline(self._cost_guide_distribution,
    #     self._potential_function,
    #     self._pool_size,
    #     max_iterations=self._max_iterations,
    #     max_outer_iterations=self._max_outer_iterations)
    # min_cost = out['min_cost']
    # start_cost = out['init_cost']
    # rmsd = torch.sqrt(torch.sum((torch.Tensor(start_cost) - torch.Tensor(min_cost))**2) / torch.Tensor(start_cost).shape[0])
    # debug_data = {
    #     'initial_energy': out['einit'],
    #     'final_energy': out['efinal'],
    #     'attempts': out['min_attempts'],
    #     'rmsd': rmsd
    # }
    cost_guides, potentials = analog_minimize.run_pipeline(self._cost_guide_distribution,
        self._potential_function,
        self._pool_size,
        max_iterations=self._max_iterations,
        max_outer_iterations=self._max_outer_iterations)

    print(cost_guides)
    print(potentials)
    return cost_guides, potentials