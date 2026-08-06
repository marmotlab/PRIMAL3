"""
Run from the repo root with the torch/ray env, e.g.:
    /home/patrick/anaconda3/envs/MAPF/bin/python 32_size_maps/random_1000/run_the_instances.py
    /home/patrick/anaconda3/envs/MAPF/bin/python 32_size_maps/random_1000/run_the_instances.py \
        --svg_save_dir 32_size_maps/random_1000/svgs --num_cases 5
"""

import os
import sys
import argparse

# --- make the repo importable and give absolute paths, so this script works no
# --- matter the current working directory (and on ray worker processes too). ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ray
import torch
from model import Model
from mapf_gym import MapfGym
from alg_parameters import *
from util import OneEpPerformance, make_gif, init_colors
import numpy as np
import pickle

CASES_PATH = os.path.join(
    REPO_ROOT, "32_size_maps", "random_1000",
    "{}length_{}agents_{}density.pth".format(
        EnvParameters.WORLD_SIZE[0], EnvParameters.N_AGENTS, EnvParameters.OBSTACLE_PROB[1]),
)
RESTORE_PATH = os.path.join(
    REPO_ROOT, "models", "primal3",
    "primal3_v22_pibt_inherit_v217-06-261404", "26427392")


def _rgb_to_hex(rgb):
    """init_colors() returns rgb floats in [0, 1]; turn one into '#rrggbb'."""
    return "#{:02x}{:02x}{:02x}".format(*(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb))


def make_animation_svg(obstacle_map, goals, agent_traj, n_agents, file_name,
                       cell=20, step_dur=0.3):
    """Write an animated SVG of one episode (the MapfGym analogue of POGEMA's
    `env.save_animation`, which the HMAGAT runner uses).

    obstacle_map : HxW array, any non-zero entry is an obstacle (MapfGym stores
                   -1 for obstacles, 0 for free space).
    goals        : list of (row, col), one static goal per agent.
    agent_traj   : list over timesteps; each entry is a list of (row, col), one
                   per agent. Agents animate smoothly between consecutive steps.
    n_agents     : number of agents (for the colour wheel, matching _render()).
    cell         : pixel size of one grid cell.
    step_dur     : seconds spent moving between two consecutive timesteps.
    """
    obstacle_map = np.asarray(obstacle_map)
    h, w = obstacle_map.shape
    width, height = w * cell, h * cell
    colours = init_colors(n_agents)  # {agent_idx+1: rgb}, same mapping as _render

    n_steps = len(agent_traj)
    total_dur = max(step_dur * max(n_steps - 1, 1), step_dur)

    def cx(col):
        return col * cell + cell / 2.0

    def cy(row):
        return row * cell + cell / 2.0

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect x="0" y="0" width="{}" height="{}" fill="#ffffff"/>'.format(width, height),
    ]

    # obstacles
    for (r, c), val in np.ndenumerate(obstacle_map):
        if val != 0:
            parts.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#1a1a1a"/>'.format(
                c * cell, r * cell, cell, cell))

    radius = cell * 0.4

    # goals: hollow circle outlined in the agent's colour (static)
    for i, (r, c) in enumerate(goals):
        col_hex = _rgb_to_hex(colours[i + 1])
        parts.append(
            '<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="none" '
            'stroke="{}" stroke-width="{:.1f}"/>'.format(
                cx(c), cy(r), radius, col_hex, max(cell * 0.1, 1.0)))

    # agents: filled circle that animates along its trajectory
    for i in range(n_agents):
        col_hex = _rgb_to_hex(colours[i + 1])
        xs = ";".join("{:.1f}".format(cx(agent_traj[t][i][1])) for t in range(n_steps))
        ys = ";".join("{:.1f}".format(cy(agent_traj[t][i][0])) for t in range(n_steps))
        start = agent_traj[0][i]
        parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="{}">'.format(
            cx(start[1]), cy(start[0]), radius, col_hex))
        if n_steps > 1:
            parts.append(
                '<animate attributeName="cx" values="{}" dur="{:.2f}s" '
                'repeatCount="indefinite"/>'.format(xs, total_dur))
            parts.append(
                '<animate attributeName="cy" values="{}" dur="{:.2f}s" '
                'repeatCount="indefinite"/>'.format(ys, total_dur))
        parts.append('</circle>')

    parts.append('</svg>')

    with open(file_name, "w") as f:
        f.write("\n".join(parts))


@ray.remote(num_cpus=1)
def test_model(num_episode, svg_save_dir=None):
    net_path_checkpoint = os.path.join(RESTORE_PATH, "net_checkpoint.pkl")
    # during the testing, we are gonna to use cpu
    net_dict = torch.load(net_path_checkpoint, map_location=torch.device('cpu'), weights_only=False)
    test_device = torch.device('cpu')
    test_model = Model(0, test_device)
    test_model.network.load_state_dict(net_dict['model'])
    test_model.network.eval()
    with torch.no_grad():
        success_status, makespan, num_reached = test_one_case(
            model=test_model, test_episode=num_episode, svg_save_dir=svg_save_dir)
    return success_status, makespan, num_reached


def test_one_case(model, test_episode, svg_save_dir=None):
    """
        model is the evaluation model - the trained network
    """
    print(f"current_episode is {test_episode}")
    # env_info[0] = world with agents; env_info[1] = world with goals; env_info[2] = obstacle map;
    # env_info[3] = agents;            env_info[4] = agents goals;     env_info[5] = nodes_obs;
    # env_info[6] = num_agents
    with open(CASES_PATH, 'rb') as f:
        env_info = pickle.load(f)

    fully_arrived = 0
    one_ep_step = 0
    num_reached = 0
    oneEpisodePerformance = OneEpPerformance()

    # Build the env directly from the (solvable) case. We bypass MapfGym.__init__
    # -- whose random throwaway initialisation crashes in astar_4 at this scale --
    # because `replicate` rebuilds every attribute __init__ would have set.
    env = MapfGym.__new__(MapfGym)
    env.replicate(-1 * env_info[test_episode][2], env_info[test_episode][3], env_info[test_episode][4])

    save_svg = svg_save_dir is not None
    agent_traj = []  # per-timestep list of [(row, col), ...], one entry per agent

    done = False
    while not done:
        obs, vector, svo, conflict_index, same_index, msg = env.getAllObservations()
        if save_svg:
            agent_traj.append([a.getPos('mat') for a in env.agentList])
        actions, pre_block, _, _, ps, svo_output = model.evaluate(
            obs, vector, svo, conflict_index, same_index, msg, num_agent=env.n_agents
        )
        actionStatus, fixedActions = env.getActionStatus(actions, svo_output, ps=ps)
        oneEpisodePerformance.invalid += len(env.getStaticColl(actionStatus))
        svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(fixedActions, actionStatus)
        oneEpisodePerformance.block += np.sum(blockings)
        oneEpisodePerformance.numLeaveGoal += np.sum(leaveGoals)
        oneEpisodePerformance.numCollide += np.sum(numCollide)
        oneEpisodePerformance.numStep += 1
        for i in range(env.n_agents):
            if (pre_block[i] < 0.5) == blockings[:, i]:
                oneEpisodePerformance.wrongBlocking += 1
        oneEpisodePerformance.episodeReward += np.sum(baseRewards)
        goalsReached, truelly_done = env.jointStep(fixedActions)
        if truelly_done or ((oneEpisodePerformance.numStep + 1) % EnvParameters.EPISODE_LEN == 0):
            done = True
        oneEpisodePerformance.maxGoals = max(oneEpisodePerformance.maxGoals, np.sum(goalsReached))
        if np.sum(goalsReached) == env.n_agents:
            fully_arrived = 1
        one_ep_step = oneEpisodePerformance.numStep
        num_reached = oneEpisodePerformance.maxGoals

        if done and save_svg:
            agent_traj.append([a.getPos('mat') for a in env.agentList])  # final positions
            goals = [a.getGoal('mat') for a in env.agentList]
            os.makedirs(svg_save_dir, exist_ok=True)
            status = "success" if truelly_done else "fail"
            out = os.path.join(svg_save_dir, "anim_{}_{}.svg".format(test_episode, status))
            make_animation_svg(env.obstacleMap, goals, agent_traj, env.n_agents, out)
            print(f"wrote {out}")

    return fully_arrived, one_ep_step, num_reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PRIMAL3 on the random_1000 instance set.")
    parser.add_argument("--svg_save_dir", type=str, default=None,
                        help="If set, save an animated SVG per case as <dir>/anim_<case>.svg")
    parser.add_argument("--num_cases", type=int, default=200,
                        help="Number of cases to run (default: 200).")
    args = parser.parse_args()

    total_success = 0
    total_step = 0
    total_reach = 0
    # PYTHONPATH so ray worker processes can import the repo modules too.
    ray.init(num_cpus=25, runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}})
    num_runs = args.num_cases
    results = ray.get([test_model.remote(i, args.svg_save_dir) for i in range(num_runs)])
    for result in results:
        if result[0] == 1:
            total_success += result[0]
        total_step = total_step + result[1]
        total_reach = total_reach + result[2]
    print(f"Map type is random; env size is {EnvParameters.WORLD_SIZE[0]}; num_agent is {EnvParameters.N_AGENTS}; obs_prob is {EnvParameters.OBSTACLE_PROB[1]}")
    print("The max steps is", EnvParameters.EPISODE_LEN)
    print(f"success rate is: {total_success / num_runs}")
    print("the average step is: ", total_step / num_runs)
    print("the reach rate is: ", total_reach / (EnvParameters.N_AGENTS * num_runs))


    # How to run this script:
    #   1. Activate the torch/ray env (or whatever env you used to train the model).
    #   2. From the repo root, run:
    #        python run_the_instances.py run_the_instances.py --svg_save_dir svgs  --num_cases 100
    #   3. The `svgs` directory will contain one animated SVG per case, named `anim_<case>_success.svg` or `anim_<case>_fail.svg`.
