import io
import time
from typing import Collection, Optional, Sequence

import torch
import random
import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


def _minimize(
        cost_guide,
        potential_function,
        max_iterations: int):
    """Minimize energy via LBFGS."""
    ret = {}

    ret["init_cost"] = cost_guide
    tensor_cost_guide = torch.Tensor(cost_guide).to(device)

    ret["einit"]     = potential_function.forward(tensor_cost_guide).item()
    ret["min_cost"], ret["efinal"] = potential_function.optimize(tensor_cost_guide, max_iterations)
    return ret


def _run_one_iteration(
        *,
        cost_guide,
        potential_function,
        max_iterations: int,
        max_attempts: int):
    """Runs the minimization pipeline.

    Args:
        pdb_string: A pdb string.
        max_iterations: An `int` specifying the maximum number of L-BFGS iterations.
        A value of 0 specifies no limit.
        tolerance: kcal/mol, the energy tolerance of L-BFGS.
        stiffness: kcal/mol A**2, spring constant of heavy atom restraining potential.
        restraint_set: The set of atoms to restrain.
        max_attempts: The maximum number of minimization attempts.
        use_gpu: Whether to run on GPU.
        exclude_residues: An optional list of zero-indexed residues to exclude from
            restraints.

    Returns:
        A `dict` of minimization info.
    """
    # Assign physical dimensions.
    start = time.time()
    minimized = False
    attempts = 0
    while not minimized and attempts < max_attempts:
        attempts += 1
        try:
            print(f"Minimizing analog, attempt {attempts} of {max_attempts}.")
            ret = _minimize(cost_guide, potential_function, max_iterations=max_iterations)
            minimized = True
        except Exception as e:  # pylint: disable=broad-except
            print(e)

    if not minimized:
        raise ValueError(f"Minimization failed after {max_attempts} attempts.")

    ret["opt_time"] = time.time() - start
    ret["min_attempts"] = attempts
    return ret


def run_pipeline(
        cost_guide_distribution,
        potential_function,
        pool_size: int = 5,
        max_outer_iterations: int = 1,
        max_iterations: int = 0,
        max_attempts: int = 10,
        checks: bool = True):
    """Run iterative amber relax.

    Successive relax iterations are performed until all violations have been
    resolved. Each iteration involves a restrained Amber minimization, with
    restraint exclusions determined by violation-participating residues.

    Args:
        prot: A cost guide protein to be relaxed.
        stiffness: kcal/mol A**2, the restraint stiffness.
        use_gpu: Whether to run on GPU.
        max_outer_iterations: The maximum number of iterative minimization.
        place_hydrogens_every_iteration: Whether hydrogens are re-initialized
            prior to every minimization.
        max_iterations: An `int` specifying the maximum number of L-BFGS steps
            per relax iteration. A value of 0 specifies no limit.
        tolerance: kcal/mol, the energy tolerance of L-BFGS.
            The default value is the OpenMM default.
        restraint_set: The set of atoms to restrain.
        max_attempts: The maximum number of minimization attempts per iteration.
        checks: Whether to perform cleaning checks.
        exclude_residues: An optional list of zero-indexed residues to exclude from
            restraints.

    Returns:
        out: A dictionary of output values.
    """
    violations = np.inf
    iteration = 0

    cost_guides  = cost_guide_distribution.sample(size=pool_size)
    potentials = [potential_function.forward(torch.Tensor(cost_guide).to(device)).item() for cost_guide in cost_guides]
    cost_guides  = [cost_guide for _, cost_guide in sorted(zip(potentials, cost_guides))]

    # randomly sample one cost guide
    sampled_index = random.choice(range(pool_size))
    potential_cost_guide = cost_guides[sampled_index]

    while violations > 0 and iteration < max_outer_iterations:
        if iteration >= 1:
            # add noise
            potential_cost_guide = cost_guide_distribution.add_noise(potential_cost_guide)

        ret = _run_one_iteration(
            cost_guide=potential_cost_guide,
            potential_function=potential_function,
            max_iterations=max_iterations,
            max_attempts=max_attempts)

        potential_cost_guide  = ret["min_cost"]
        potential = ret["efinal"]

        # check violation
        if potential_cost_guide not in cost_guides:
            cost_guides.append(potential_cost_guide)
            potentials.append(potential)

        print(f"Iteration completed: Einit {ret['einit']} Efinal {ret['efinal']} Time {ret['opt_time']} s ")
        iteration += 1
    
    return cost_guides, potentials