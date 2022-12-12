
from typing import Union

import shutil
import inspect
import json

import gym
from gym.spaces import Box

from ai4water.backend import os, np, pd, plt
from ai4water.utils.utils import dateandtime_now
from ai4water.backend import easy_mpl as ep

from ._swat import SWAT


class SWATSingleReservoir(gym.Env):
    """
    parameters
    -----------
    state_names :
        names of columns from output.rch file to be used as states

    >>> swat_env = SWATSingleReservoir(
    ... swat_path= 'path/to/swat/files',
    ... backup_path= 'path/to/swat/files/as/backup',
    ... start_day=5, delta=3, lookback=3)
    ...
    >>> model = PPO(policy="MlpPolicy", env=swat_env)
    ...
    ... # Stop training if there is no improvement after more than 3 evaluations
    >>> stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    >>> eval_callback = EvalCallback(swat_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    ...
    >>> trained_ppo = model.learn( total_timesteps=1_000)
    ...
    >>> swat_env.save_results()
    >>> swat_env.save_plots()

    """
    def __init__(
            self,
            swat_path:str,
            backup_path:str,
            weir_codes:dict,
            start_day:int = 5,
            delta:int = 5,
            end_day:int = 365,
            lookback:int = 4,
            downstream_rch_id: int = 144,
            reservoir_id: int = 134,
            year:int = 2017,
            state_names:Union[str, list] = "FLOW_INcms"
    ):

        self.swat_path = swat_path
        self.backup_path = backup_path
        self.weir_codes = weir_codes

        self.start_day = start_day
        self.end_day = end_day
        self.weir_num = str(reservoir_id).rjust(3, '0')

        if isinstance(state_names, str):
            state_names = [state_names]
        assert isinstance(state_names, list)
        self.state_names = state_names

        src = os.path.join(backup_path, f"00{self.weir_num}0000.day")
        dst = os.path.join(swat_path, f"00{self.weir_num}0000.day")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "file.cio")
        dst = os.path.join(swat_path, "file.cio")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.rch")
        dst = os.path.join(swat_path, "output.rch")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.wql")
        dst = os.path.join(swat_path, "output.wql")
        shutil.copy(src, dst)

        self.reservoir_id = reservoir_id
        self.year = year
        self.downstream_rch_id = downstream_rch_id
        self.delta = delta
        self.lookback = lookback
        self.swat = SWAT(swat_path, weir_codes=weir_codes)
        self.swat.change_nbyr(1)
        self.swat.change_iyr(2017)
        self.swat.change_start_day(start_day - self.lookback)
        self.swat.change_end_day(start_day + delta)
        self.swat.write_outflow(self.reservoir_id, np.full(10, 0.0))

        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(len(state_names),), dtype=np.float32)

        self.terminal = False

        if len(state_names)==1:
            self.state_t0 = 0
        else:
            self.state_t0 = [0 for _ in range(len(self.state_names))]

        self.state = self.state_t0
        self.reward = 0

        self.num_episodes = 0
        self.total_steps = 0
        self.day = self.start_day
        self.total_steps = 0

        self.step_rewards = []  # reward at each step
        self.step_actions = []
        self.step_states = []
        self.step_chla = []
        self.steps_in_episode = 0  # steps in a single episode

        self.ep_mean_rewards = []  # mean reward for the episode
        self.ep_sum_rewards = []
        self.ep_steps = []  # total steps in each episode

        # following are not reset
        self.step_rewards_total = []  # step rewards for all episodes
        self.step_actions_total = []
        self.step_states_total = []
        self.step_chla_total = []

        self.path = os.path.join(os.getcwd(), "results", f"{dateandtime_now()}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.save_config()

    @staticmethod
    def get_reward(chla):
        chla = float(chla)
        return -2.5e-5 * chla**2

    def step(self, action):

        action = action * 100

        # feed action to SWAT
        chla, state = self.run_swat(action, self.day)

        # calculate reward
        self.reward = self.get_reward(chla)

        self.day += self.delta
        self.total_steps += 1
        self.steps_in_episode += 1

        self.state = state

        self.step_rewards.append(self.reward)
        self.step_actions.append(action)
        self.step_states.append(state)
        self.step_chla.append(chla)

        self.step_rewards_total.append(self.reward)
        self.step_states_total.append(self.state)
        self.step_actions_total.append(action)
        self.step_chla_total.append(chla)

        if self.day>= self.end_day:
            self.terminal = True

        return self.state, self.reward, self.terminal, {}

    def render(self, mode="human", close=False):
        return self.state

    def reset(self):
        self.terminal = False
        self.state =self.state_t0

        self.ep_mean_rewards.append(np.mean(self.step_rewards))
        self.ep_sum_rewards.append(np.sum(self.step_rewards))
        self.ep_steps.append(self.steps_in_episode)

        self.num_episodes += 1

        self.step_rewards = []
        self.step_actions = []
        self.step_states = []
        self.step_chla = []

        print(f"episode: {self.num_episodes}, steps in episode: {self.steps_in_episode}  total steps: {self.total_steps}")

        self.steps_in_episode = 0
        self.day = self.start_day

        return self.state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        """
        Even if an error is encountered during ``fit``, the results, report and config
        must be saved.

        """
        if exc_type:
            print(f"{exc_type} occured")

        self.save_results(model=None)
        self.save_plots()

        return

    def run_swat(self, outflow:float, day, constituent="ALGAE_INppm"):
        """runs the SWAT model for a single step with the outflow
        # ALGAE_INppm, CHLA_INkg
        >>> swat_env = SWATSingleReservoir(start_day=5, delta=3, lookback=3)
        >>> a,b = swat_env.run_swat(50, 51)
        """
        # swat output at the start

        # write outflow in .day file
        outflow_df = self.swat.get_weir_outflow(self.reservoir_id)
        self.swat.write_outflow(self.reservoir_id, np.full(len(outflow_df), outflow))

        # run swat
        self.swat.change_start_day(day - self.lookback)
        self.swat.change_end_day(day + self.delta)
        self.swat(executable='swat_683_silent')

        # read new chl-a concentration
        rch_out = self.swat.read_rch(self.downstream_rch_id, self.year)
        wql_out = self.swat.read_wql_output(self.downstream_rch_id)
        #a = jday_to_date(year, day)

        chla = wql_out.loc[:, constituent].mean()
        # convert from ppm to mg/m3
        chla = 0.0409 * chla * 893.5

        return chla, rch_out.loc[:, self.state_names].mean().values.tolist()

    def config(self)->dict:
        _config = dict()
        _config['reward_func'] = inspect.getsource(self.get_reward)
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

    def save_results(self, model=None):

        n = len(self.step_states_total)

        pd.DataFrame(
            np.concatenate([
                np.array(self.step_actions_total).reshape(-1, 1),
                np.array(self.step_rewards_total).reshape(-1, 1),
                np.array(self.step_chla_total).reshape(-1, 1),
                np.array(self.step_states_total).reshape(n, len(self.state_names)),
                 ], axis=1
            ),
            columns=["actions", "rewards", "chla", *self.state_names]
        ).to_csv(os.path.join(self.path, "results.csv"))

        pd.DataFrame(
            np.concatenate([
                np.array(self.step_actions).reshape(-1,1),
                np.array(self.step_rewards).reshape(-1, 1),
                np.array(self.step_states).reshape(-1, len(self.state_names)),
                np.array(self.step_chla).reshape(-1, 1),
            ], axis=1),
            columns=["actions", "rewards", "chla", *self.state_names]
        ).to_csv(os.path.join(self.path, "last_episode.csv"))

        pd.DataFrame(
            np.array(self.ep_steps),
            columns=["steps_per_episode"]
        ).to_csv(os.path.join(self.path, "steps_per_episode.csv"))

        pd.DataFrame(
            np.concatenate([
                np.array(self.ep_mean_rewards).reshape(-1,1),
                np.array(self.ep_sum_rewards).reshape(-1,1)
            ], axis=1),
            columns=["mean", "sum"]
        ).to_csv(os.path.join(self.path, "episode_rewards.csv"))

        if model is not None:
            model_attrs = {}
            for attr in [
                "n_epochs",
                "n_steps",
                "batch_size",
                "observation_space",
                "action_space",
                "num_timesteps",
                "gamma",
                "gae_lambda",
                "learning_rate",
            ]:
                model_attrs[attr] = str(getattr(model, attr))
            model_attrs['policy_name'] = model.policy._get_name()
            with open(os.path.join(self.path, "model_attrs.json"), "w") as fp:
                json.dump(model_attrs, fp, indent=4, sort_keys=True)

            pd.DataFrame(
                model.rollout_buffer.values,
                columns=["rollout_buffer_values"]
            ).to_csv(os.path.join(self.path, "rollout_buff_values.csv"))
            ep.plot(model.rollout_buffer.values,
                 title="rollout_buffer_values",
                 xlabel="Time Steps (Days)",
                 show=False
                 )
            plt.savefig(os.path.join(self.path, "rollout_buff_values"))

            model.save(os.path.join(self.path, "model"))
        return

    def save_plots(self):

        plt.close('all')
        ep.plot(self.step_rewards,
             title="Rewards during last episode",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "rew_last_ep"))

        plt.close('all')
        ep.plot(self.step_chla,
             title="Chla during last episode",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "chla_last_ep"))

        plt.close('all')
        ep.plot(self.step_actions,
             title="Actions during last episode",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "act_last_ep"))

        plt.close('all')
        ep.plot(self.step_chla_total,
             title="Chla during whole training",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "chla_whole_train"))

        plt.close('all')
        ep.plot(self.step_states_total,
             title="State during whole training",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "state_whole_train"))

        plt.close('all')
        ep.plot(self.step_actions_total, linewidth=0.3,
             title="Actions during whole training",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "act_whole_train"))

        plt.close('all')
        ep.plot(self.step_rewards_total, linewidth=0.3,
             title="Rewards during whole training",
             xlabel="Time Steps (Days)", show=False)
        plt.savefig(os.path.join(self.path, "rew_whole_train"))

        plt.close('all')
        ep.plot(self.ep_sum_rewards, '--.',
             title="Sum of rewards after each Episode",
             xlabel="Episodes", show=False)
        plt.savefig(os.path.join(self.path, "sum_rew_ep"))

        plt.close('all')
        ep.plot(self.ep_mean_rewards,
             title="Average of rewards after each Episode",
             xlabel="Episodes", show=False)
        plt.savefig(os.path.join(self.path, "avg_rew_ep"))
        plt.close('all')

        return
