import datetime
import math

""" Hyperparameters """


class EnvParameters:
    N_AGENTS = 50  # 50/100/150/200/250/300
    N_ACTIONS = 5
    N_SVOs = 10     # previously as 10
    EPISODE_LEN = 512  # maximum episode length in training 
    FOV_SIZE = 9   # previously set to 9
    FOV_Heuristic = 5
    WORLD_SIZE = (32, 32)    # 20, 30
    OBSTACLE_PROB = (0.19, 0.2)
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1
    CONFLICT_OVERLAP_DECAY = 0.9  # 0.9
    SAME_OVERLAP_DECAY = 0.5      # 0.5
    IMPORTANCE_SVO = 2
    # Weight of the selfishness SVO bias in the PIBT shielding priority.
    # priority = pibt_age + PIBT_SVO_WEIGHT * (N_SVOs - 1 - svo_avg) + dist_base.
    # 0 -> distance/age-only PIBT; larger -> stronger social yielding.
    PIBT_SVO_WEIGHT = 1
    PIBT_AGE_WEIGHT = 1
    PIBT_SVO_MAX_AGENTS = 1000000
    PIBT_LARGE_AGENT_SVO_WEIGHT = 0
    PIBT_SHIELD_INPUT_TYPE = "probs" 

    # Expert running time limit
    EXPERT_LIMIT = 0.03
    EXPERT_TIMESTEP = 0.2

    # ---- OOD input clipping for cross-scale evaluation ----
    # The checkpoint was trained on random maps of size 20~30 with <=32 agents.
    # On larger evaluation maps (e.g. 72x72 / 1000 agents) the scale-dependent
    # scalar inputs exceed anything seen in training. If CLIP is True, mapf_gym
    # clips them to the training-time maxima below; if False, observations are
    # produced exactly as before.
    CLIP = True
    TRAIN_MAX_WORLD_SIZE = 30  # largest map edge used in training
    TRAIN_MAX_N_AGENTS = 32    # largest crowd used in training
    CLIP_VECTOR_DIST_MAX = (TRAIN_MAX_WORLD_SIZE - 1) * math.sqrt(2)
    CLIP_MSG_DIST2GOAL_MAX = 2 * (TRAIN_MAX_WORLD_SIZE - 1)
    CLIP_MSG_NUM_CUTVERTEX_MAX = 8
    CLIP_MSG_HARD_BLOCKED_MAX = TRAIN_MAX_N_AGENTS - 1
    CLIP_MSG_SOFT_BLOCKED_MAX = TRAIN_MAX_N_AGENTS - 1

    # ---- fast path extraction for large-scale evaluation ----
    # True: shortest paths are read off each agent's cached goal-BFS map
    # (O(path length)) inside get_agent_neighbor_and_overlap and getAgentsMsgs
    # instead of running astar_4 per agent per step. Paths stay equally short,
    # but tie-breaking between equal-length routes can differ from A*, which
    # may slightly shift SVO neighbor selection and numCutvertex/softBlocked.
    # False: the original astar_4 pipeline everywhere.
    FAST_PATHS = True


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 32  # number of processes
    N_MAX_STEPS = 3e7  # 3e7 for official training, 3e5 for debug
    N_STEPS = 2 ** 8  # number of time steps per process per data collection
    MINIBATCH_SIZE = int(2 ** 8)

    # curriculum: stage1 uses N_AGENTS, stage2+ uses a random choice from LARGE_N_AGENTS
    STAGE2_START_STEPS = 5e6   # step at which agent count scales up
    # 5e6 for official training, 5e4 for debug
    LARGE_N_AGENTS = [16, 24, 32]  # options for the large-agent curriculum

    # three stages training
    STAGE1_MAX_STEPS = 2.3e7  # 2.3e7 for official training, 2.3e5 for debug


class NetParameters:
    NET_SIZE = 512
    SVO_C_SIZE = 512
    NUM_CHANNEL = 11  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 4  # [dx, dy, d total, action t-1]
    MSG_LEN = 7


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = True
    RESTORE_PATH = './models/primal3/primal3_v22_pibt_inherit_v217-06-261404/26427392'
    WANDB = True
    TENSORBOARD = False
    TXT_WRITER = True
    ENTITY = 'chengyanghe'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'primal3'
    EXPERIMENT_NAME = 'primal3_v22_pibt_inherit_v2'
    EXPERIMENT_NOTE = 'precedence in observation and action added'
    SAVE_INTERVAL = 5e5  # interval of saving model0
    BEST_INTERVAL = 0  # interval of saving model0 with the best performance
    GIF_INTERVAL = 5e5  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TEST_GIFS_PATH = './test_gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME



all_args = dict()

for i in dir(EnvParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(EnvParameters, i)

for i in dir(TrainingParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(TrainingParameters, i)

for i in dir(NetParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(NetParameters, i)

for i in dir(SetupParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(SetupParameters, i)
