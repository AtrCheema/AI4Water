
from typing import Union, List

import gym
from gym.spaces import Box


class FlowCalibration(gym.Env):
    """
    A gym environment which can be used to calibrate streamflow
    by dynamic optimization of parameters using HSP2 (Hydrologic Simulation
    Program Python)

    Parameters
    -----------
    target_rch :
        the rchres id for which the flow ``q`` is to be calibrated
    state_params :
        the parameters to consider as state. These can be any time series
        generated for the particular hru.
    cal_params :
        the parameters to calibrate
    hrus_of_params :
        the ids of hrus from which ``cal_params`` are considered.
    hsp2_timestep : int (default=None)
        timestep at which hsp2 is run
    cal_params_timestep : int (default=None)
        timestep at which the calibrated parameters are changed. The HSP2 will
        be run for this time-step at every RL iteration/step.
    """
    def __init__(
            self,
            target_rch:Union[str, int],
            state_params:List[str],
            cal_params:Union[str, List[str]],
            hrus_of_params:Union[str, List[str]] = "all",
            hsp2_timestep:int = None,
            cal_params_timestep:int = None
    ):
        self.rch = target_rch
        self.state_params = state_params
        self.cal_params = cal_params
        self.hrus = hrus_of_params
        self.hsp2_timestep = hsp2_timestep
        self.cal_params_timestep = cal_params_timestep

        self.state = check_state_params(state_params)

    def reward(self)->float:
        # compare observed and predicted flows and return the reward
        raise NotImplementedError

    def step(self, action):
        """
        parameters
        --------
        action :
            parameters of HSPF as suggested by agent
        """

        # change parameters in hdf file
        self.change_paras_in_hdf(action)

        # run the hsp2
        self.run_hsp2()

        # calculate reward
        self.reward = self.reward()

        # calculate state

        # determine if it is terminal state or not

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        return

    def run_hsp2(self):
        # change start and end time of hsp2

        # run the hsp2
        return

    def change_paras_in_hdf(self, parameters):
        return


def check_state_params(state_params):
    return state_params
