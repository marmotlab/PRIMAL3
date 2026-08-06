import numpy as np
import ray
import time
import torch

from alg_parameters import *
from mapf_gym import MapfGym
from model import Model
from util import OneEpPerformance, BatchValues, PerfDict
from expert_guidance import parse_expert_conf, solve_lacam_once, get_expert_action_from_cached_paths

_TIMING_PRINTED = False


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)

    def run(self, weights, curr_glb_steps, num_agents=None):
        return episodeRun(model=self.local_model, weights=weights, eval=False, curr_glb_steps=curr_glb_steps, num_agents=num_agents)
    
def _collect_hard_instance_batch(model, obstacle_map, initial_positions, initial_goals, lacam_paths):
    """Replay a hard instance from scratch guided by LaCAM paths and collect training data."""
    env = MapfGym()
    env.replicate(obstacle_map, initial_positions, initial_goals)

    mb = BatchValues()
    svo_rewardList = []
    action_rewardList = []
    rewardList = []
    doneList = []
    done = False

    max_path_len = max(len(p) for p in lacam_paths)
    n_steps = min(max_path_len - 1, EnvParameters.EPISODE_LEN)
    if n_steps <= 0:
        return None

    for step_idx in range(n_steps):
        obs, vector, svo, conflict_index, same_index, msg = env.getAllObservations()

        mb.observations.append(obs)
        mb.vector.append(vector)
        mb.svo.append(svo)
        mb.conflict_index.append(conflict_index)
        mb.same_index.append(same_index)
        mb.msg.append(msg)

        actions, ps, values, pre_block, _, svo_output = model.step(obs, vector, svo, conflict_index, same_index, msg)

        lacam_actions, _ = get_expert_action_from_cached_paths(lacam_paths, step_idx)
        actionStatus, fixedActions = env.getActionStatus(lacam_actions, svo_output)

        fixedExptPs = np.full((len(fixedActions), EnvParameters.N_ACTIONS), 0)
        fixedExptPs[np.arange(len(fixedActions)), fixedActions.astype(int)] = 1

        svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = \
            env.calculateReward(fixedActions, actionStatus)

        mb.values.append(values)
        mb.ps.append(ps)
        mb.actions.append(actions)
        mb.blocking.append(blockings)
        mb.svo_exe.append(env.getAgentsSVOexe())
        mb.trainValid.append(env.getTrainValid(fixedActions, actionStatus))
        mb.expert_action.append(fixedActions)
        mb.expert_ps.append(fixedExptPs)

        svo_rewardList.append(svo_post_rewards)
        action_rewardList.append(action_post_rewards)
        rewardList.append(baseRewards)
        doneList.append(done)

        goalsReached, done = env.jointStep(fixedActions)
        if done:
            break

    actual_n = len(mb.actions)
    mb.observations = np.concatenate(mb.observations, axis=0)
    mb.vector = np.concatenate(mb.vector, axis=0)
    mb.svo = np.concatenate(mb.svo, axis=0)
    mb.svo_exe = np.asarray(mb.svo_exe, dtype=np.int64)
    mb.conflict_index = np.concatenate(mb.conflict_index, axis=0)
    mb.same_index = np.concatenate(mb.same_index, axis=0)
    mb.msg = np.concatenate(mb.msg, axis=0)
    svo_rewardList = np.concatenate(svo_rewardList, axis=0)
    action_rewardList = np.concatenate(action_rewardList, axis=0)
    rewardList = np.concatenate(rewardList, axis=0)
    mb.values = np.squeeze(np.concatenate(mb.values, axis=0), axis=-1)
    mb.actions = np.asarray(mb.actions, dtype=np.int64)
    mb.ps = np.stack(mb.ps)
    doneList = np.asarray(doneList, dtype=np.bool_)
    mb.trainValid = np.stack(mb.trainValid)
    mb.blocking = np.concatenate(mb.blocking, axis=0)
    mb.expert_action = np.asarray(mb.expert_action, dtype=np.int64)
    mb.expert_ps = np.stack(mb.expert_ps)

    last_obs, last_vec, last_svo, last_ci, last_si, last_msg = env.getAllObservations()
    last_values = np.squeeze(model.value(last_obs, last_vec, last_svo, last_ci, last_si, last_msg))

    mb_advs_svo = np.zeros_like(svo_rewardList)
    mb_advs_action = np.zeros_like(action_rewardList)
    mb_advs = np.zeros_like(rewardList)
    last_gaelam_svo = last_gaelam_action = last_gaelam = 0

    for t in reversed(range(actual_n)):
        if t == actual_n - 1:
            next_nonterminal = 1.0 - done
            next_values = last_values
        else:
            next_nonterminal = 1.0 - doneList[t + 1]
            next_values = mb.values[t + 1]

        delta_svo = np.subtract(np.add(svo_rewardList[t], TrainingParameters.GAMMA * next_nonterminal * next_values), mb.values[t])
        delta_action = np.subtract(np.add(action_rewardList[t], TrainingParameters.GAMMA * next_nonterminal * next_values), mb.values[t])
        delta = np.subtract(np.add(rewardList[t], TrainingParameters.GAMMA * next_nonterminal * next_values), mb.values[t])

        mb_advs_svo[t] = last_gaelam_svo = np.add(delta_svo, TrainingParameters.GAMMA * TrainingParameters.LAM * next_nonterminal * last_gaelam_svo)
        mb_advs_action[t] = last_gaelam_action = np.add(delta_action, TrainingParameters.GAMMA * TrainingParameters.LAM * next_nonterminal * last_gaelam_action)
        mb_advs[t] = last_gaelam = np.add(delta, TrainingParameters.GAMMA * TrainingParameters.LAM * next_nonterminal * last_gaelam)

    mb.returns_svo = np.add(mb_advs_svo, mb.values)
    mb.returns_action = np.add(mb_advs_action, mb.values)
    mb.returns = np.add(mb_advs, mb.values)

    return mb


def episodeRun(model:Model, weights = None, eval = False, curr_glb_steps=0, num_agents=None):
    global _TIMING_PRINTED
    assert(eval or weights!=None)
    if num_agents is None:
        if curr_glb_steps < TrainingParameters.STAGE2_START_STEPS:
            num_agents = EnvParameters.N_AGENTS
        else:
            num_agents = int(np.random.choice(TrainingParameters.LARGE_N_AGENTS))
    _t0 = time.perf_counter()
    env = MapfGym(num_agents=num_agents)
    _graph_build_time = time.perf_counter() - _t0
    _t_obs_total = _t_fwd_total = _t_expert_total = _t_env_total = 0.0
    _rollout_start = time.perf_counter()

    """run multiple steps and collect data for reinforcement learning"""
    with torch.no_grad():

        if not eval:
            model.set_weights(weights)
            mb = BatchValues()
        else:
            episodeFrames = []

        performanceDict = PerfDict()
        oneEpisodePerformance = OneEpPerformance()

        svo_rewardList = list()
        action_rewardList = list()
        rewardList = list()
        doneList = list()
        done = False

        # Save initial instance state for potential hard-instance retraining
        initial_obstacle_map = env.obstacleMap.copy()
        initial_positions = env.get_positions()
        initial_goals = env.get_goals()

        expert_paths_cache = None
        expert_cache_start_step = None
        expert_cache_solvable = True

        for _ in range(TrainingParameters.N_STEPS):

            _ts = time.perf_counter()
            obs, vector, svo, conflict_index, same_index, msg = env.getAllObservations()
            _t_obs_total += time.perf_counter() - _ts

            if eval:
                episodeFrames.append(env._render())
                actions, pre_block, _, _, ps, svo_output = model.evaluate(obs, vector, svo, conflict_index, same_index, msg, num_agent=env.n_agents)
            else:
                mb.observations.append(obs)
                mb.vector.append(vector)
                mb.svo.append(svo)
                mb.conflict_index.append(conflict_index)
                mb.same_index.append(same_index)
                mb.msg.append(msg)

                _ts = time.perf_counter()
                actions, ps, values, pre_block, _, svo_output = model.step(obs, vector, svo, conflict_index, same_index, msg, num_agent=env.n_agents)
                _t_fwd_total += time.perf_counter() - _ts

                # create empty actions and ps
                empty_actions = np.zeros_like(actions)
                empty_ps = np.zeros_like(ps)

                if curr_glb_steps < TrainingParameters.STAGE1_MAX_STEPS:
                    mb.expert_action.append(empty_actions)
                    mb.expert_ps.append(empty_ps)
                else:
                    curr_ep_step = oneEpisodePerformance.numStep
                    if curr_ep_step < 100:
                        # Early episode: call lacam each step for uncertain agents only
                        expert_guided_actions, _ = parse_expert_conf(actions, ps, env.obstacleMap, np.array(env.get_positions()), np.array(env.get_goals()), curr_ep_step)
                    else:
                        # Late episode: solve lacam once at step 100, then index into cached paths
                        if expert_paths_cache is None:
                            print("Running LACAM for the first time at step 100 of episode, caching results for future steps...")
                            expert_paths_cache, expert_cache_solvable = solve_lacam_once(env.obstacleMap, np.array(env.get_positions()), np.array(env.get_goals()))
                            expert_cache_start_step = curr_ep_step
                        if expert_cache_solvable and expert_paths_cache is not None:
                            path_idx = curr_ep_step - expert_cache_start_step
                            expert_guided_actions, _ = get_expert_action_from_cached_paths(expert_paths_cache, path_idx)
                        else:
                            # Both 30s and 60s failed: use model actions to advance env,
                            # then reset cache so lacam is retried from the new positions next step.
                            expert_guided_actions = actions
                            expert_paths_cache = None
                            expert_cache_start_step = None
                            expert_cache_solvable = True
                    actionExpStatus, fixedExptActions = env.getActionStatus(expert_guided_actions, svo_output)
                    # convert fixedExptActions to fixedExptPs
                    fixedExptPs = np.full((len(fixedExptActions), EnvParameters.N_ACTIONS), 0)
                    fixedExptPs[np.arange(len(fixedExptActions)), fixedExptActions.astype(int)] = 1
                    mb.expert_ps.append(fixedExptPs)
                    mb.expert_action.append(fixedExptActions)

                mb.values.append(values)
                mb.ps.append(ps)
                doneList.append(done)
            

            _ts = time.perf_counter()
            actionStatus, fixedActions = env.getActionStatus(actions, svo_output, ps=ps)
            _t_env_total += time.perf_counter() - _ts

            oneEpisodePerformance.invalid += len(env.getStaticColl(actionStatus))
            
            # svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(actions, actionStatus)

            if curr_glb_steps < TrainingParameters.STAGE1_MAX_STEPS:
                svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(fixedActions, actionStatus)
            else:
                if eval:
                    svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(fixedActions, actionStatus)
                else:
                    svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(fixedExptActions, actionExpStatus)

            oneEpisodePerformance.block+=np.sum(blockings)
            oneEpisodePerformance.numLeaveGoal += np.sum(leaveGoals)
            oneEpisodePerformance.numCollide+= np.sum(numCollide)
            oneEpisodePerformance.numStep+=1

            for i in range(env.n_agents):
                    if (pre_block[i] < 0.5) == blockings[:, i]:
                        oneEpisodePerformance.wrongBlocking += 1
        
            if not eval:
                mb.svo_exe.append(env.getAgentsSVOexe())
                # mb.trainValid.append(env.getTrainValid(actions, actionStatus))
                if curr_glb_steps < TrainingParameters.STAGE1_MAX_STEPS:
                    mb.trainValid.append(env.getTrainValid(fixedActions, actionStatus))
                else:
                    mb.trainValid.append(env.getTrainValid(fixedExptActions, actionExpStatus))
                mb.actions.append(actions)
                mb.blocking.append(blockings)
                svo_rewardList.append(svo_post_rewards)
                action_rewardList.append(action_post_rewards)
                rewardList.append(baseRewards)
            
            oneEpisodePerformance.episodeReward += np.sum(baseRewards)

            # goalsReached, done = env.jointStep(fixedActions)
            if curr_glb_steps < TrainingParameters.STAGE1_MAX_STEPS:
                goalsReached, done = env.jointStep(fixedActions)
            else:
                if not eval:
                    # the state will updated according to the ExptActions
                    goalsReached, done = env.jointStep(fixedExptActions)
                else:
                    # the state will updated according to the model actions
                    goalsReached, done = env.jointStep(fixedActions)

            if(done or ((oneEpisodePerformance.numStep+1)%EnvParameters.EPISODE_LEN==0)):
                    done = True

            if oneEpisodePerformance.numStep == EnvParameters.EPISODE_LEN // 2:
                performanceDict.Half_goals.append(np.sum(goalsReached))

            oneEpisodePerformance.maxGoals = max(oneEpisodePerformance.maxGoals, np.sum(goalsReached))


            if done:
                performanceDict.__update__(oneEpisodePerformance, np.sum(goalsReached), n_agents=env.n_agents)
                if eval:
                    episodeFrames.append(env._render()) #append frame to gif
                    break
                else:
                    oneEpisodePerformance = OneEpPerformance()
                    _ts = time.perf_counter()
                    env = MapfGym(num_agents=num_agents)
                    _graph_build_time += time.perf_counter() - _ts
                    done = True
                    expert_paths_cache = None
                    expert_cache_start_step = None
                    expert_cache_solvable = True

        _rollout_total = time.perf_counter() - _rollout_start
        if not eval and not _TIMING_PRINTED:
            _TIMING_PRINTED = True
            N = TrainingParameters.N_STEPS
            print(f"\n[TIMING v20] steps={N}:")
            print(f"  rollout/env step time (total):   {_rollout_total*1000:.0f} ms  ({_rollout_total/N*1000:.2f} ms/step)")
            print(f"  graph/map construction time:     {_graph_build_time*1000:.0f} ms")
            print(f"  obs building (getAllObservations):{_t_obs_total*1000:.0f} ms  ({_t_obs_total/N*1000:.2f} ms/step)")
            print(f"  model forward time:              {_t_fwd_total*1000:.0f} ms  ({_t_fwd_total/N*1000:.2f} ms/step)")
            print(f"  expert/LaCAM time:               {_t_expert_total*1000:.0f} ms")
            print(f"  env step (getActionStatus etc):  {_t_env_total*1000:.0f} ms  ({_t_env_total/N*1000:.2f} ms/step)")
            obs_bytes = EnvParameters.N_AGENTS * NetParameters.NUM_CHANNEL * EnvParameters.FOV_SIZE * EnvParameters.FOV_SIZE * 4
            print(f"  obs size per step:               {obs_bytes//1024} KB")

        # Collect initial instance state for post-training (eval only, after 2e7 steps)
        initial_instance = None
        if eval and curr_glb_steps >= 2e7 and oneEpisodePerformance.numStep > 150:
            initial_instance = (initial_obstacle_map, initial_positions, initial_goals)

        if not eval:
            mb.observations = np.concatenate(mb.observations, axis=0)
            mb.vector = np.concatenate(mb.vector, axis=0)
            mb.svo = np.concatenate(mb.svo, axis=0)
            mb.svo_exe = np.asarray(mb.svo_exe, dtype=np.int64)
            mb.conflict_index = np.concatenate(mb.conflict_index, axis=0)
            mb.same_index = np.concatenate(mb.same_index, axis=0)
            mb.msg = np.concatenate(mb.msg, axis=0)
            svo_rewardList = np.concatenate(svo_rewardList, axis=0)
            action_rewardList = np.concatenate(action_rewardList, axis=0)
            rewardList = np.concatenate(rewardList, axis=0)
            mb.values = np.squeeze(np.concatenate(mb.values, axis=0), axis=-1)
            mb.actions = np.asarray(mb.actions, dtype=np.int64)
            mb.ps = np.stack(mb.ps)
            doneList = np.asarray(doneList, dtype=np.bool_)
            mb.trainValid = np.stack(mb.trainValid)
            mb.blocking = np.concatenate(mb.blocking, axis=0)

            # ----------------- confidence boosting labels ----------------
            mb.expert_action = np.asarray(mb.expert_action, dtype=np.int64)
            mb.expert_ps = np.stack(mb.expert_ps)
            # -------------------------------------------------------------

            last_values  = np.squeeze(model.value(obs, vector, svo, conflict_index, same_index, msg))

            # calculate advantages
            mb_advs_svo = np.zeros_like(svo_rewardList)
            mb_advs_action = np.zeros_like(action_rewardList)
            mb_advs = np.zeros_like(rewardList)
            last_gaelam_svo = 0
            last_gaelam_action = 0
            last_gaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0 - done
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - doneList[t + 1]
                    next_values= mb.values[t + 1]

                delta_svo = np.subtract(np.add(svo_rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                               next_values), mb.values[t])
                delta_action = np.subtract(np.add(action_rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                                  next_values), mb.values[t])
                delta = np.subtract(np.add(rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                            next_values), mb.values[t])

                mb_advs_svo[t] = last_gaelam_svo = np.add(delta_svo, TrainingParameters.GAMMA * TrainingParameters.LAM
                                                          * next_nonterminal * last_gaelam_svo)
                mb_advs_action[t] = last_gaelam_action = np.add(delta_action,
                                                                TrainingParameters.GAMMA * TrainingParameters.LAM
                                                                * next_nonterminal * last_gaelam_action)
                mb_advs[t] = last_gaelam = np.add(delta, TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)
            mb.returns_svo = np.add(mb_advs_svo, mb.values)
            mb.returns_action = np.add(mb_advs_action, mb.values)
            mb.returns = np.add(mb_advs, mb.values)
            
    if eval:
        return performanceDict, episodeFrames, initial_instance
    else:
        return mb, performanceDict
