import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional

from alg_parameters import *

import sys, os
sys.path.append(os.path.abspath("lacam3"))
from build.bindings import lacam as lacam
from astar_4 import astar_4

# for confidence part
import copy

"""
Confidence Moduale for Expert Guidance
"""

def compute_entropy(policy_probs):
    # get the entropy of the policy probabilities
    entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-10), axis=-1)
    return entropy

def find_definitive_pc(nn_pc_entropy):
    """The threshold is set as 0.46371241 bc it is the entropy of pc [0.9, 0.025, 0.025, 0.025, 0.025]"""
    threshold = 0.46371241
    # the indices are the agents whose pc is not good / stable enough
    indices = np.where(nn_pc_entropy > threshold)[0]
    return indices

def get_expert_action(obstacle_map, starts, goals):
    """
    obstacle map: same as self.obstacle_map in mapf_gym
    starts:       np.array with shape (EnvParameters.N_AGENTS, 2)
    goals:        np.array with shape (EnvParameters.N_AGENTS, 2)
    """

    # print(obstacle_map)
    # print("starts", starts)
    # print("goals", goals)

    def find_difference_index(tuple1, tuple2):
        diff = (tuple2[0] - tuple1[0], tuple2[1] - tuple1[1])
        diff_map = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (0, -1): 3,
            (-1, 0): 4
        }
        return diff_map.get(diff, -1)  # if there is something wrong, return -1
    n_agents = len(starts)
    expert_actions = -1 * np.ones(n_agents)

    # print("before process: ", expert_actions)

    solvable_flag_lacam = True

    """
    expert_paths[i][0]: agent i's current position
    expert_paths[i][1]: agent i's goal position
    If agent i stay on its goal, expert_paths[i] = [(x_goal, y_goal)]
    """
    # covnert the obstacle_map to the format that lacam3 can understand
    obstacle_map = -1 * obstacle_map
    _time_limits = [EnvParameters.EXPERT_LIMIT, 0.5, 1.0, 10.0, 30.0]
    expert_paths = []
    for _limit in _time_limits:
        expert_paths = lacam.solve(obstacle_map.tolist(), starts, goals, _limit)
        if len(expert_paths) >= n_agents:
            break
    else:
        solvable_flag_lacam = False

    # only process the expert paths if solvable_flag_lacam is True
    if solvable_flag_lacam:
        for i in range(n_agents):
            # print("current agent: ", i, "expert_paths[i]: ", expert_paths[i])
            if len(expert_paths[i]) == 0:
                return -1 * np.ones(n_agents)
            if len(expert_paths[i]) == 1:
                expert_actions[i] = 0
            else:
                # use next_pos - curr_pos to determine the expert action
                expert_actions[i] = find_difference_index(expert_paths[i][0], expert_paths[i][1])

        # print("after process: ", expert_actions)

        # according to the expert_actions, build expert_ps
        epsilon = 0.025   # this "epsilon" control the prob of the other actions
        expert_ps = np.full((len(expert_actions), EnvParameters.N_ACTIONS), epsilon)
        expert_ps[np.arange(len(expert_actions)), expert_actions.astype(int)] = 1 - 4 * epsilon
    else:
        epsilon = 0.025
        expert_ps = np.full((len(expert_actions), EnvParameters.N_ACTIONS), epsilon)

    return expert_actions, expert_ps, solvable_flag_lacam

def parse_expert_conf(nn_actions, nn_ps, obstacle_map, starts, goals, curr_ep_step):

    expert_guided_actions = copy.deepcopy(nn_actions)
    expert_guided_ps = copy.deepcopy(nn_ps)
    expert_actions, expert_ps, lacam_solvable = get_expert_action(obstacle_map, starts, goals)
    if lacam_solvable:
        if curr_ep_step < 100:   #   This is a heuristic value, we can change it later
            # if the current episode step is less than 70, we use expert guidance for the agents not confident enough
            nn_output_entropy = compute_entropy(nn_ps)
            unstable_agent_idx = find_definitive_pc(nn_output_entropy)
            if len(unstable_agent_idx) != 0:
                expert_guided_actions[unstable_agent_idx] = expert_actions[unstable_agent_idx]
                expert_guided_ps[unstable_agent_idx] = expert_ps[unstable_agent_idx]
        else:
            # if the current episode step is more than 70, we use expert guidance for all agents
            # we assume that most of the agents are reached theirs goas already
            expert_guided_actions = expert_actions 
            expert_guided_ps = expert_ps
    else:
        # if the expert path is not solvable, we use the current model's actions and ps
        print("Expert path not found in 10s, using current model's actions and ps.")
        expert_guided_actions = nn_actions
        expert_guided_ps = nn_ps

    return expert_guided_actions, expert_guided_ps


def solve_lacam_once(obstacle_map, starts, goals):
    """Call lacam.solve with two attempts: first 30s, then 60s.
    Returns (expert_paths, solvable_flag).
    expert_paths[i] is the full path list for agent i from its current position to its goal.
    If both attempts fail, returns (None, False).
    """
    om = (-1 * obstacle_map).tolist()
    starts_list = list(starts)
    goals_list = list(goals)

    n_agents = len(starts)
    # First attempt: 30s
    expert_paths = lacam.solve(om, starts_list, goals_list, 30.0)
    if len(expert_paths) >= n_agents:
        return expert_paths, True

    # Second attempt: 60s
    print("LaCAM 30s attempt failed, retrying with 60s.")
    expert_paths = lacam.solve(om, starts_list, goals_list, 60.0)
    if len(expert_paths) >= n_agents:
        return expert_paths, True

    print("LaCAM 60s attempt also failed, falling back to model actions this step.")
    return None, False


def get_expert_action_from_cached_paths(expert_paths, path_idx):
    """Extract expert actions from cached lacam paths at the given step offset.
    path_idx=0 means use expert_paths[i][0]->expert_paths[i][1] (first step from cache start).
    """
    def find_difference_index(t1, t2):
        diff = (t2[0] - t1[0], t2[1] - t1[1])
        diff_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
        return diff_map.get(diff, -1)

    n_agents = len(expert_paths)
    expert_actions = np.zeros(n_agents)
    for i in range(n_agents):
        path = expert_paths[i]
        if len(path) == 0 or path_idx >= len(path) - 1:
            expert_actions[i] = 0  # stay (at or past goal)
        else:
            expert_actions[i] = find_difference_index(path[path_idx], path[path_idx + 1])

    epsilon = 0.025
    expert_ps = np.full((n_agents, EnvParameters.N_ACTIONS), epsilon)
    expert_ps[np.arange(n_agents), expert_actions.astype(int)] = 1 - 4 * epsilon
    return expert_actions, expert_ps


# -------------------------primal3 v6d expert guidance module ------------------------- #

Pos = Tuple[int, int]
Path = List[Pos]

def astar_paths_lengths(world: np.ndarray, agent_coords: List[Pos], agent_goals: List[Pos]):
    paths = []
    lengths = []
    for i in range(len(agent_coords)):
        start = tuple(agent_coords[i])
        goal  = tuple(agent_goals[i])
        if start != goal:
            path, _ = astar_4(world, start, goal)
        else:
            path = [goal]
        paths.append(path)
        lengths.append(len(path) - 1)
    return lengths, paths

def collapse_trailing_repeats(path):
    """Collapse trailing repeats of the last state: [..., last, last, last] -> [..., last]."""
    if not path:
        return 0, []
    last = path[-1]
    k = 0
    i = len(path) - 1
    while i >= 0 and path[i] == last:
        k += 1
        i -= 1
    collapsed = path[:len(path) - k] + [last]
    return k, collapsed

def lacam_paths_lengths(obstacle_map: np.ndarray, agent_coords, agent_goals, astar,
                        init_limit: float = EnvParameters.EXPERT_LIMIT,
                        step: float = EnvParameters.EXPERT_TIMESTEP,
                        max_limit: float = 5.0):
    """
    Like astar_paths_lengths: return (lengths, paths).
    paths are LaCAM paths with trailing repeated goal collapsed.
    """
    # keep your original LaCAM convention conversion
    om = (-1 * obstacle_map).tolist()

    limit = float(init_limit)
    expert_paths = lacam.solve(om, agent_coords, agent_goals, limit)

    n_agents = len(agent_coords)
    while len(expert_paths) < n_agents:
        limit += step
        if limit > max_limit:
            # fallback: stay at start
            paths = [[tuple(s)] for s in agent_coords]
            lengths = [astar[i] for i in range(n_agents)]
            print(f"LaCAM failed to find paths for all agents within {max_limit} seconds. Returning fallback paths.")
            return lengths, paths
        expert_paths = lacam.solve(om, agent_coords, agent_goals, limit)

    paths = []
    lengths = []
    for i in range(n_agents):
        _, collapsed = collapse_trailing_repeats(expert_paths[i])
        path = [tuple(x) for x in collapsed]
        paths.append(path)
        lengths.append(len(path) - 1)

    return lengths, paths

def get_supervised_label(astar_lengths, lacam_lengths, graph_mask):
    """
    Return a binary label: 1 if A* path is strictly shorter than LaCAM path, else 0.
    """
    labels = []
    for astar_len, lacam_len in zip(astar_lengths, lacam_lengths):
        label = 1 if lacam_len < astar_len else 0
        labels.append(label)
    return labels


