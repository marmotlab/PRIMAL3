import random

import imageio
import numpy as np
import torch
import wandb

import cv2, math
from matplotlib.colors import hsv_to_rgb

from alg_parameters import *
from enum import Enum

class Status(Enum):

    REACH_GOAL = 3
    LEAVE_GOAL = 2
    VALID = 1 # 0: action executed
    REPEAT_ACTION = -1
    STATIC_COLLISION = -3 # out of boundaries or collision with obstacles
    AGENT_COLLSION = -2

class BatchValues:
    def __init__(self):
        self.observations = list()
        self.vector = list()
        self.svo = list()
        self.svo_exe = list()
        self.conflict_index = list()
        self.same_index = list()
        self.msg = list()
        self.returns_svo = list()
        self.returns_action = list()
        self.returns = list()
        self.values = list()
        self.actions = list()
        self.ps = list()
        # -------------------- confidence labels -----------------
        self.expert_action = list()
        self.expert_ps = list()
        # -------------------------------------------------------
        self.trainValid = list()
        self.blocking = list()
    def __repr__(self) -> str:
        temp = ""
        for i in dir(self):
            if not i.startswith("__"):
                temp+=i+":"+str(getattr(self, i))+" "

        return temp



class OneEpPerformance():
    def __init__(self):
        self.episodeReward = 0
        self.numStep = 0
        self.invalid = 0
        self.block = 0
        self.numLeaveGoal = 0
        self.numCollide = 0
        self.wrongBlocking = 0
        self.maxGoals = 0
    def __repr__(self) -> str:
        temp = ""
        for i in dir(self):
            if not i.startswith("__"):
                temp+=i+":"+str(getattr(self, i))+" "
        return temp


class PerfDict():
    def __init__(self):
        self.Reward = list()
        self.Valid_rate = list()
        self.Episode_length = list()
        self.Num_block = list()
        self.Num_leave_goal = list()
        self.Final_goals = list()
        self.Half_goals = list()
        self.Block_accuracy = list()
        self.Max_goals = list()
        self.Num_collide = list()

    def __update__(self, oneEpPerf:OneEpPerformance, numOnGoal, n_agents=EnvParameters.N_AGENTS):
        self.Reward.append(oneEpPerf.episodeReward)
        self.Valid_rate.append(
            ((oneEpPerf.numStep*n_agents) - oneEpPerf.invalid)/ (oneEpPerf.numStep*n_agents))
        self.Episode_length.append(oneEpPerf.numStep)
        self.Num_block.append(oneEpPerf.block)
        self.Num_leave_goal.append(oneEpPerf.numLeaveGoal)
        self.Final_goals.append(numOnGoal)
        self.Block_accuracy.append(
            ((oneEpPerf.numStep*n_agents) - oneEpPerf.wrongBlocking)/(oneEpPerf.numStep*n_agents))
        self.Max_goals.append(oneEpPerf.maxGoals)
        self.Num_collide.append(oneEpPerf.numCollide)
    
    def __repr__(self) -> str:
        temp = ""
        for i in dir(self):
            if not i.startswith("__"):
                # print(i, getattr(self, i))
                temp+=i+":"+str(getattr(self, i))+" "
        return temp


class Loss():
    def __init__(self):
        self.all_loss = 0
        self.policy_loss = 0
        self.policy_entropy = 0
        self.critic_loss = 0
        self.valid_loss = 0
        self.blocking_loss = 0
        self.clipfrac = 0
        self.grad_norm = 0
        self.advantage = 0
    def __repr__(self) -> str:
        temp = ""
        for i in dir(self):
            if not i.startswith("__"):
                temp+=i+":"+str(getattr(self, i))+" "
        return temp



def getFreeCell(world):
    
    listOfFree = np.swapaxes(np.where(world==0), 0,1)
    np.random.shuffle(listOfFree)
    return (listOfFree[0][0], listOfFree[0][1])


def returnAsType(arr, type):
    if(type=='np'): # numpy array
        return arr
    elif(type=='mat'): # to be used directly as a cell of matrix
        return (arr[0], arr[1])
    else:
        raise Exception("Invalid Type as input")


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def getMeanLoss(mb_loss):
    meanLoss = Loss()
    for i in dir(meanLoss):
        if not i.startswith('__'):
            temp = list()
            for j in mb_loss:
                temp.append(getattr(j,i))
                setattr(meanLoss, i, np.nanmean(temp, axis=0))
    return meanLoss

def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    wandb.log({'Perf_greedy_eval/'+i: getattr(performance_dict, i)}, step=step)

        else:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    wandb.log({'Perf_random_eval/'+i: getattr(performance_dict, i)}, step=step)

    else:
        meanLoss = getMeanLoss(mb_loss)
        for i in dir(performance_dict):
            if not i.startswith('__'):
                wandb.log({'Perf/'+i: getattr(performance_dict, i)}, step=step)

        for i in dir(meanLoss):
            if not i.startswith('__'):
                if i == 'grad_norm':
                    wandb.log({'Grad/' + i: getattr(meanLoss, i)}, step=step)
                else:
                    wandb.log({'Loss/' + i: getattr(meanLoss, i)}, step=step)


def make_gif(images, file_name):
    """record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")

    
def init_colors(n_agents=EnvParameters.N_AGENTS):
    """the colors of agents and goals"""
    c = {a + 1: hsv_to_rgb(np.array([a / float(n_agents), 1, 1])) for a in range(n_agents)}
    c[0] = [1,1,1]
    c[-1] = [0,0,0]
    c[-2] = [0.5,0.5,0.5]
    return c

def getRectPoints(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return np.array([base, [base[0]+scale-1, base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def pixelForText(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return [int(math.floor(base[0]+scale*1/4)), int(math.floor(base[1]+scale*3/4))]


def getCenter(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return [int(math.floor(base[0]+scale/2)), int(math.floor(base[1]+scale/2))]

def getTriPoints( coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return  np.array([[int(math.floor(base[0]+scale/2)), base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def renderWorld(scale=20, world = np.zeros(1),agents=[], goals=[], svoOrder = [], n_agents=EnvParameters.N_AGENTS):
    size = world.shape

    screen_height = scale*size[0]
    screen_width = scale*size[1]

    colours = init_colors(n_agents)

    scene = np.zeros([screen_height, screen_width, 3])

    for coord,val in np.ndenumerate(world):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=colours[val])


    for val,coord in enumerate(goals):
        cv2.circle(scene, getCenter(coord=coord, scale=scale), math.floor(scale/2)-1, colours[val+1], -1)
        # cv2.putText(scene, str(val+1), pixelForText(coord, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/40, (0,0,0), int(scale/20))


    for val,coord in enumerate(agents):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=colours[val+1])
        cv2.putText(scene, str(svoOrder[val]), pixelForText(coord, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/40, (0,0,0), int(scale/20))

    scene = scene*255
    scene = scene.astype(dtype='uint8')
    return scene


def symmetric_normalize(A):
    # Compute the degree matrix D for each adjacency matrix in the batch
    degree = torch.sum(A, dim=-1)
    D = torch.diag_embed(degree)

    # Compute D^(-1/2) for each matrix in the batch
    D_inv_sqrt = torch.inverse(torch.sqrt(D))

    # Compute the symmetrically normalized adjacency matrix for each matrix in the batch
    normalized_adjacency = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
    return normalized_adjacency
