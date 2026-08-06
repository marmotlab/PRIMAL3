import ray
import torch
from model import Model
from mapf_gym import MapfGym
from alg_parameters import *
from util import OneEpPerformance, make_gif
import numpy as np
import os
import pickle


@ray.remote(num_cpus=1)
def test_model(num_episode):
    # load the fully trained model
    # restore_path = 'models/primal3/primal3_maze/final'
    restore_path = 'models/primal3/primal3_v22_pibt_inherit_v217-06-261404/26427392'
    net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
    # during the testing, we are gonna to use cpu
    # load the trained model
    net_dict = torch.load(net_path_checkpoint, map_location=torch.device('cpu'), weights_only=False)
    test_device = torch.device('cpu')
    test_model = Model(0, test_device)
    test_model.network.load_state_dict(net_dict['model'])
    test_model.network.eval()
    with torch.no_grad():
        success_status, makespan, num_reached = test_one_case(model=test_model, test_episode=num_episode)
    return success_status, makespan, num_reached


def test_one_case(model, test_episode):
    """
        The env_size should be a tuple as env_size = (20, 20)
        num_agents = 50, 100, 150, 200, 250, 300
        model is the evaluation model - the trained network
        device: run the execution on cpu or gpu
        greedy: choose the action greedy or randomly
    """
    print(f"current_episode is {test_episode}")
    # load the parameters from args, i.e., the item in tests
    # env_info[0] = world with agents; env_info[1] = world with goals; env_info[2] = obstacle map;
    # env_info[3] = agents;            env_info[4] = agents goals;     env_info[5] = nodes_obs;
    # env_info[6] = num_agents
    # general tests
    with open('./32_size_maps/32_32_0.2/{}length_{}agents_{}density.pth'.format(EnvParameters.WORLD_SIZE[0], EnvParameters.N_AGENTS, EnvParameters.OBSTACLE_PROB[1]), 'rb') as f:
        env_info = pickle.load(f)

    fully_arrived = 0
    one_ep_step = 0
    num_reached = 0
    # episodeFrames = []
    oneEpisodePerformance = OneEpPerformance()

    # Build the env directly from the (solvable) saved case. Calling
    # MapfGym.__init__ here first generated a throwaway RANDOM map/agent layout
    # and ran A* on it before replicate() ever loaded the case; that random
    # layout is not guaranteed solvable, so it crashed with "No Path Exists"
    # at random. replicate() rebuilds every attribute __init__ would have set.
    env = MapfGym.__new__(MapfGym)
    env.replicate(-1 * env_info[test_episode][2], env_info[test_episode][3], env_info[test_episode][4])

    done = False
    while not done:
        obs, vector, svo, conflict_index, same_index, msg = env.getAllObservations()
        # episodeFrames.append(env._render())
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

        # if done:
        #     episodeFrames.append(env._render())
        #     if not os.path.exists(RecordingParameters.TEST_GIFS_PATH):
        #         os.makedirs(RecordingParameters.TEST_GIFS_PATH)
        #     # print("frames:", len(episode_frames))
        #     images = np.array(episodeFrames[:-1])
        #     name = str(test_episode)
        #     make_gif(images, RecordingParameters.TEST_GIFS_PATH + "/" + name + '.gif')

    return fully_arrived, one_ep_step, num_reached



if __name__ == "__main__":
    # init some metric
    total_success = 0
    total_step = 0
    total_reach = 0
    # how many env should test in parallel
    ray.init(num_cpus=25)
    num_runs = 200
    results = ray.get([test_model.remote(i) for i in range(num_runs)])
    for result in results:
        if result[0] == 1:
            total_success += result[0]
        total_step = total_step + result[1]
        total_reach = total_reach + result[2]
    print(f"Map type is maze; env size is {EnvParameters.WORLD_SIZE[0]}; num_agent is {EnvParameters.N_AGENTS}; obs_prob is {EnvParameters.OBSTACLE_PROB[1]}")
    print("The max steps is", EnvParameters.EPISODE_LEN)
    print(f"success rate is: {total_success / num_runs}")
    print("the average step is: ", total_step / num_runs)
    print("the reach rate is: ", total_reach / (EnvParameters.N_AGENTS * num_runs))


