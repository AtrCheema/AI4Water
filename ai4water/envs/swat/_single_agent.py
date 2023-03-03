
from typing import Union, Tuple

import shutil
import inspect
import json

import gym
from gym.spaces import Box

from ai4water.backend import os, np, pd, plt
from ai4water.utils.utils import dateandtime_now
from ai4water.backend import easy_mpl as ep

from ._swat import SWAT, jday_to_date
from ._global_vars import RCH_COL_MAP


RCH_COL_MAP_ = {v:k for k,v in RCH_COL_MAP.items()}


class SWATSingleReservoir(gym.Env):
    """
    gym environment which can be used for single weir optimization/control.

    parameters
    -----------
    swat_path : str
        path where SWAT model results/files are located. The files in this folder
        are changed when SWAT is run.
    backup_path : str
        same as swat_path but the files in this folder are not changed. The output files are
        copied from backup_path to swat_path so that the user has a backup of original
        results.
    weir_location : int
        reservoir/reach on which weir is located. The outflow from this
        reservoir will be controlled by using .day file and
        the reservoir parameters in downstream to this reservoir
        will be read from corresponding .wql and .rch files and used
        used as ``state``.
    start_day : int
        julian day to start. The actual start day of SWAT will be start_day - lookback
    end_day : int
    lookback : int
        The simulation duration of SWAT for each time-step of RL in an episode
    year : int
        simulation year
    state_names : str/list
        names of columns/variables from output.rch file to be used as states.
        If you want to consider meteorological parameters as state as well,
        consider using ``add_met_data_to_state``.

    Examples
    --------
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
            weir_location:int,
            start_day:int = 5,
            delta:int = 5,
            end_day:int = 365,
            lookback:int = 4,
            year:int = 2017,
            state_names:Union[str, list] = "FLOW_INcms"
    ):

        self.swat_path = swat_path
        self.backup_path = backup_path

        self.start_day = start_day
        self.end_day = end_day
        self.weir_num = str(weir_location).rjust(3, '0')

        self.state_names = self._process_states(state_names)
        self.num_states = len(self.state_names)
        self.state_names_rch = state_names.copy()
        self.use_met_states = False

        # the files needs to be copied from backup folder before initializing
        # the SWAT class
        self._copy_backup_files()

        # initialize the class which interacts with SWAT model
        self.swat = SWAT(swat_path)
        self.swat.change_num_sim_years(1)
        self.swat.change_sim_start_yr(year)
        self.swat.change_start_day(start_day - lookback)
        self.swat.change_end_day(start_day + delta)
        self.swat.write_outflow(weir_location, np.full(self.swat.sim_len(), 0.0))
        self.swat.customize_sub_output(1)

        if "WTMPdegc" in state_names:
            cols = [RCH_COL_MAP_[col] for col in state_names]
        else:
            cols = 0
        self.swat.customize_rch_output(cols)
        self.swat.customize_hru_output(1)

        self.weir_location = weir_location
        self.year = year
        self.downstream_rch_id =  self.swat.downstream_rch_to_rch(weir_location)
        self.delta = delta
        self.lookback = lookback
        self.start_date = jday_to_date(year, start_day)
        self.end_date = jday_to_date(year, end_day)

        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(self.num_states,), dtype=np.float32)

        self.terminal = False

        # state at the start
        if self.num_states==1:
            self.state_t0 = 0
        else:
            self.state_t0 = [0 for _ in range(self.num_states)]

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
    def _process_states(state_names):
        if isinstance(state_names, str):
            state_names = [state_names]
        assert isinstance(state_names, list)
        return state_names.copy()

    def _copy_backup_files(self):
        """copies .day, .cio, and output files from backup folder to
        swath_path"""
        fname = f"00{self.weir_num}0000.day"
        src = os.path.join(self.backup_path, fname)

        if not os.path.exists(src):
            raise FileExistsError(f"File {fname} is not found in {self.backup_path}")
        dst = os.path.join(self.swat_path, fname)
        shutil.copy(src, dst)

        for file in ["file.cio", "output.rch", "output.wql", "output.rsv",
                     "output.sub", "output.hru", "output.std", "input.std"]:
            src = os.path.join(self.backup_path, file)
            dst = os.path.join(self.swat_path, file)
            shutil.copy(src, dst)

        return

    def add_met_data_to_state(
            self,
            parameters:Union[list, str]
    )->None:
        """
        add the meteorological parameters such as precipitation, humidity,
        wind speed, air temperature for the state.
        This method overwrites the following class variable.
            ``observation_space``
             ``state_t0``
             ``num_states``
             ``state_names``
             ``use_met_states``

        This method creates ``met_paras`` class variable

        parameters
        -----------
        parameters :
            Allowed values are
                pcp
                min_temp
                max_temp
                sol_rad
                wind_speed
                rel_hum

        """
        allowed =  ['pcp', 'min_temp', 'max_temp', 'sol_rad', 'wind_speed', 'rel_hum']

        if not isinstance(parameters, list):
            parameters = [parameters]
        assert all([para in allowed for para in parameters])

        self.num_states += len(parameters)
        self.observation_space = Box(low=-1, high=1, shape=(self.num_states,), dtype=np.float32)
        self.state_t0 = [0 for _ in range(self.num_states)]
        self.state_names += parameters
        self.use_met_states =True

        self.met_paras = self.get_met_paras()[parameters]

        return

    def get_met_paras(self)->pd.DataFrame:
        # todo we are using met data for sub-basins and considering it for reach
        pcp = self.swat.precip_for_sub(self.downstream_rch_id).loc[self.start_date:self.end_date]
        hum = self.swat.rel_hum_for_sub(self.downstream_rch_id).loc[self.start_date:self.end_date]
        temp = self.swat.temp_for_sub(self.downstream_rch_id).loc[self.start_date:self.end_date]
        wind = self.swat.wind_for_sub(self.downstream_rch_id).loc[self.start_date:self.end_date]
        sol_rad = self.swat.sol_rad_for_sub(self.downstream_rch_id).loc[self.start_date:self.end_date]
        df = pd.concat([pcp, hum, temp, wind, sol_rad], axis=1)
        df.columns = ['pcp', 'rel_hum', 'max_temp', 'min_temp', 'wind_speed', 'sol_rad']
        return df

    def met_state_at_t(self, time)->list:
        """returns meteorological parameters at given time as list"""
        return self.met_paras.loc[time].values.tolist()

    @staticmethod
    def get_reward(chla):
        chla = float(chla)
        return -2.5e-5 * chla**2

    def step(self, action):

        action = action * 100

        # feed action to SWAT
        chla, state = self.run_swat(action, self.day)

        # consider meteorological data as state
        if self.use_met_states:
            state += self.met_state_at_t(jday_to_date(self.year, self.day))

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
        Even if an error is encountered during ``fit``, the results, report and
        config must be saved.

        """
        if exc_type:
            print(f"{exc_type} occured")

        self.save_results(model=None)
        self.save_plots()

        return

    def run_swat(
            self,
            outflow:float,
            day:int,
            constituent:str="ALGAE_INppm"
    )->Tuple[float, list]:
        """
        runs the SWAT model for a single step with the outflow
        # ALGAE_INppm, CHLA_INkg
        >>> swat_env = SWATSingleReservoir(start_day=5, delta=3, lookback=3)
        >>> a,b = swat_env.run_swat(50, 51)
        """

        # write outflow in .day file
        outflow_df = self.swat.get_weir_outflow(self.weir_location)
        self.swat.write_outflow(self.weir_location, np.full(len(outflow_df), outflow))

        # run swat
        self.swat.change_start_day(day - self.lookback)
        self.swat.change_end_day(day + self.delta)
        self.swat(executable='swat_683_silent')

        # read new chl-a concentration
        rch_out = self.swat.channel_output(self.downstream_rch_id, self.year)  # from output.rch file
        wql_out = self.swat.read_wql_output(self.downstream_rch_id)  # from output.wql

        chla = wql_out.loc[:, constituent].mean()
        # convert from ppm to mg/m3
        chla = 0.0409 * chla * 893.5

        return chla, rch_out.loc[:, self.state_names_rch].mean().values.tolist()

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
                np.array(self.step_states_total).reshape(n, self.num_states),
                 ], axis=1
            ),
            columns=["actions", "rewards", "chla", *self.state_names]
        ).to_csv(os.path.join(self.path, "results.csv"))

        pd.DataFrame(
            np.concatenate([
                np.array(self.step_actions).reshape(-1,1),
                np.array(self.step_rewards).reshape(-1, 1),
                np.array(self.step_states).reshape(-1, self.num_states),
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
                "max_grad_norm",
                "action_space",
                "num_timesteps",
                "gamma",
                "gae_lambda",
                "learning_rate",
                "vf_coef",
                "start_time",
                "seed",
                "sde_sample_freq",
            ]:
                model_attrs[attr] = str(getattr(model, attr, None))
            model_attrs['policy_name'] = model.policy._get_name()
            model_attrs['class_name'] = model.__class__.__name__

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
