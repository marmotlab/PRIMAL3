import numpy as np
import torch
from net_no_svo import SelfishNet


class SelfishModel(object):
    """model0 of agents"""

    def __init__(self, device):
        """initialization"""
        self.device = device
        self.selfishnet = SelfishNet().to(device)
        selfish_restore_path = './selfish_model/8/final'
        selfish_net_path_checkpoint = selfish_restore_path + "/net_checkpoint.pkl"
        selfish_net_dict = torch.load(selfish_net_path_checkpoint)
        self.selfishnet.load_state_dict(selfish_net_dict['model'])
        self.selfishnet.eval()

    def selfish_step(self, observation, vector, input_state=0):
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        new_ps_selfish, _, _, _, _, _ = self.selfishnet(observation, vector, input_state)
        new_ps_selfish = np.squeeze(new_ps_selfish.cpu().detach().numpy())
        return new_ps_selfish

    def selfish_train_step(self, observation, vector, input_state=0):
        new_ps_selfish, _, _, _, _, _ = self.selfishnet(observation, vector, input_state)
        return new_ps_selfish