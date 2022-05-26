
import json
import shutil
import inspect

import gym
from gym.spaces import Box

from ai4water.backend import os, np
from ai4water.utils.utils import dateandtime_now

from ._swat import SWAT


class SWATMultiReservoir(gym.Env):
    """
    Environment for multi-reservoir control with SWAT model as environment.

    env = SWATMultiReservoir(
        reservoir_ids = [134, 161],
        downstream_rch_ids = [136, 164]
    )

    s, r, d, i = env.step([0.5, 0.5])
    """
    def __init__(
            self,
            swat_path:str,
            reservoir_ids:list,
            downstream_rch_ids:list,
            start_day=5,
            delta=5,
            end_day=365,
            lookback=4,
            year=2017,
            max_steps=50_000
    ):
        """
        Parameters
        ----------
            swat_path : str
                path where swat files are located.
        """

        self.start_day = start_day
        self.end_day = end_day
        self.max_steps= max_steps

        assert len(reservoir_ids) == len(downstream_rch_ids)

        assert os.path.exists(swat_path)
        self.swat_path = swat_path
        src = swat_path
        basename = os.path.basename(swat_path)
        backup_path = os.path.join(os.path.dirname(swat_path), f"{basename}_backup")
        shutil.copy(src, backup_path)

        src = os.path.join(backup_path, "file.cio")
        dst = os.path.join(swat_path, "file.cio")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.rch")
        dst = os.path.join(swat_path, "output.rch")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.wql")
        dst = os.path.join(swat_path, "output.wql")
        shutil.copy(src, dst)

        for res_id in reservoir_ids:
            fname = f"00{res_id}0000.day"
            src = os.path.join(backup_path, fname)
            dst = os.path.join(swat_path, fname)
            shutil.copy(src, dst)

        self.n = len(reservoir_ids)

        # configure spaces
        self.action_space = []
        self.observation_space = []

        for i in range(self.n):
            self.action_space.append(Box(low=0, high=1, shape=(1,), dtype=np.float32))
            self.observation_space.append(Box(low=-1, high=1, shape=(1,), dtype=np.float32))

        self.reservoir_ids = reservoir_ids
        self.year = year
        self.downstream_rch_ids = downstream_rch_ids
        self.delta = delta
        self.lookback = lookback
        self.swat = SWAT(swat_path)
        self.swat.change_nbyr(1)
        self.swat.change_iyr(2017)
        self.swat.change_start_day(start_day - self.lookback)
        self.swat.change_end_day(start_day + delta)

        for res_id in self.reservoir_ids:
            self.swat.write_outflow(res_id, np.full(10, 0.0))

        self.terminal = [False, False]
        self.state = [np.array([0]).reshape(-1,), np.array([0]).reshape(-1,)]
        self.reward = [0, 0]

        self.num_episodes = 0
        self.total_steps = 0
        self.day = self.start_day

        self.step_rewards = np.full((365, self.n), np.nan)  # reward at each step
        self.step_actions = np.full((365, self.n), np.nan)
        self.step_states = np.full((365, self.n), np.nan)
        self.step_chla = np.full((365, self.n), np.nan)
        self.steps_in_episode = 0  # steps in a single episode

        self.ep_mean_rewards = np.full((max_steps, self.n), np.nan)  # mean reward for the episode
        self.ep_sum_rewards = np.full((max_steps, self.n), np.nan)
        self.ep_steps = []  # total steps in each episode

        # following are not reset
        self.step_rewards_total = np.full((max_steps, self.n), np.nan)  # step rewards for all episodes
        self.step_actions_total = np.full((max_steps, self.n), np.nan)
        self.step_states_total = np.full((max_steps, self.n), np.nan)
        self.step_chla_total = np.full((max_steps, self.n), np.nan)

        self.path = os.path.join(os.getcwd(), "results", f"{dateandtime_now()}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.save_config()

    def run_swat(self, outflows, day, constituent="CHLA_INkg"):

        # write outflow in .day file
        for idx, res_id in enumerate(self.reservoir_ids):
            outflow_df = self.swat.get_weir_outflow(res_id)
            self.swat.write_outflow(res_id, np.full(len(outflow_df), outflows[idx]))

        # run swat
        self.swat.change_start_day(day - self.lookback)
        self.swat.change_end_day(day + self.delta)
        self.swat(executable="swat_683_silent.exe")

        chlas = []
        states = []
        # read new chl-a concentration
        for res_id, rch_id in zip(self.reservoir_ids, self.downstream_rch_ids):
            rch_out = self.swat.read_rch(res_id, self.year)
            wql_out = self.swat.read_rch(rch_id, self.year)

            chlas.append(wql_out.loc[:, constituent].mean())
            state = np.array(rch_out.loc[:, "FLOW_INcms"].mean()).reshape(-1,)
            states.append(state)

        return chlas, states

    @staticmethod
    def _get_reward(chla):
        if chla>1.0:
            reward = 1/chla
        else:
            reward = 1.0
        return reward

    def get_reward(self, chlas):
        rewards = []
        for chla in chlas:
            rewards.append(self._get_reward(chla))
        return rewards

    def config(self)->dict:
        _config = dict()
        _config['reward_func'] = inspect.getsource(self.get_reward)
        _config['reward_func1'] = inspect.getsource(self._get_reward)
        _config['run_swat_func'] = inspect.getsource(self.run_swat)
        _config['init_paras'] = self._init_paras()
        return _config

    def save_config(self):
        fname = os.path.join(self.path, "config.json")
        with open(fname, "w") as fp:
            json.dump(self.config(), fp, sort_keys=True, indent=4)
        return

    def _init_paras(self) -> dict:
        """Returns the initializing parameters of this class"""
        signature = inspect.signature(self.__init__)

        init_paras = {}
        for para in signature.parameters.values():
            if para.name not in ["prefix"]:
                init_paras[para.name] = getattr(self, para.name)

        return init_paras

    def step(self, action):

        scaled_actions = []
        for act in action:
            scaled_actions.append(act * 100)

        # feed action to SWAT
        chlas, states = self.run_swat(scaled_actions, self.day)

        # calculate reward
        self.reward = self.get_reward(chlas)

        self.day += self.delta
        self.total_steps += 1
        self.steps_in_episode += 1

        self.state = states

        self.step_actions[self.steps_in_episode] = action[0][0], action[1][0]
        self.step_rewards[self.steps_in_episode] = self.reward
        self.step_chla[self.steps_in_episode] = chlas
        self.step_states[self.steps_in_episode] = states[0][0], states[1][0]

        self.step_rewards_total[self.total_steps] = self.reward
        self.step_actions_total[self.total_steps] = action[0][0], action[1][0]
        self.step_chla_total[self.total_steps] = chlas
        self.step_states_total[self.total_steps] = states[0][0], states[1][0]

        if self.day>= self.end_day:
            self.terminal = [True, True]

        return self.state, self.reward, self.terminal, {'n': [{}, {}]}

    def reset(self):

        self.terminal = [False, False]
        self.state = [np.array([0]).reshape(-1,), np.array([0]).reshape(-1,)]

        mean_rewards = np.nanmean(self.step_rewards, axis=0)
        sum_rewards = np.nansum(self.step_rewards, axis=0)
        self.ep_mean_rewards[self.num_episodes] = mean_rewards
        self.ep_sum_rewards[self.num_episodes] = sum_rewards

        self.num_episodes += 1

        self.step_rewards = np.full((365, self.n), np.nan)  # reward at each step
        self.step_actions = np.full((365, self.n), np.nan)
        self.step_states = np.full((365, self.n), np.nan)
        self.step_chla = np.full((365, self.n), np.nan)

        print(
            f"episode: {self.num_episodes}, steps in episode: {self.steps_in_episode}  total steps: {self.total_steps}")

        self.steps_in_episode = 0  # steps in a single episode
        self.day = self.start_day

        return self.state

    def render(self, mode='human'):
        raise NotImplementedError