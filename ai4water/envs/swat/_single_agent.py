
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

    swat_env = SWATSingleReservoir(start_day=5, delta=3, lookback=3)
    a,b = swat_env.run_swat(50, 51)

    model = PPO(
        policy="MlpPolicy",
        env=swat_env)


    # # Stop training if there is no improvement after more than 3 evaluations
    # #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # #eval_callback = EvalCallback(swat_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
    #
    #
    trained_ppo = model.learn(
        total_timesteps=1_000
    )

    swat_env.save_results()
    swat_env.save_plots()


    plot(model.env.envs[0].episode_returns, xlabel="Episodes",
         title="Sum of rewards during Episodes")


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
    with open(os.path.join(swat_env.path, "model_attrs.json"), "w") as fp:
        json.dump(model_attrs, fp)

    pd.DataFrame(
        model.rollout_buffer.values,
        columns=["rollout_buffer_values"]
    ).to_csv(os.path.join(swat_env.path, "rollout_buff_values.csv"))
    plot(model.rollout_buffer.values,
         title="rollout_buffer_values",
         xlabel="Time Steps (Days)"
         )
    plt.savefig(os.path.join(swat_env.path, "rollout_buff_values"))

    model.save(os.path.join(swat_env.path, "model"))
    """
    def __init__(
            self,
            start_day=5,
            delta = 5,
            end_day=365,
            lookback = 4,
            downstream_rch_id=144,
            reservoir_id=134,
            year = 2017
    ):

        self.start_day = start_day
        self.end_day = end_day

        path = os.path.join(os.getcwd(), "06_S5(2017)_no_release")
        backup_path = os.path.join(os.getcwd(), "06_S5(2017)_no_release_backup")

        src = os.path.join(backup_path, "001340000.day")
        dst = os.path.join(path, "001340000.day")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "file.cio")
        dst = os.path.join(path, "file.cio")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.rch")
        dst = os.path.join(path, "output.rch")
        shutil.copy(src, dst)

        src = os.path.join(backup_path, "output.wql")
        dst = os.path.join(path, "output.wql")
        shutil.copy(src, dst)

        self.reservoir_id = reservoir_id
        self.year = year
        self.downstream_rch_id = downstream_rch_id
        self.delta = delta
        self.lookback = lookback
        self.swat = SWAT(path)
        self.swat.change_nbyr(1)
        self.swat.change_iyr(2017)
        self.swat.change_start_day(start_day - self.lookback)
        self.swat.change_end_day(start_day + delta)
        self.swat.write_outflow(self.reservoir_id, np.full(10, 0.0))

        self.action_space = Box(low=0, high=1, shape=(1,))
        self.observation_space = Box(low=-1, high=1, shape=(1,))

        self.terminal = False
        self.state = 0
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
        if chla>0.001:
            return -1
        else:
            return +1

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
        self.state = 0

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

    def run_swat(self, outflow:float, day, constituent="ALGAE_INppm"):
        """runs the SWAT model for a single step with the outflow
        # ALGAE_INppm, CHLA_INkg
        """
        # swat output at the start
        #res_out_ini = self.swat.read_rch(downstream_rch_id, year)

        # write outflow in .day file
        outflow_df = self.swat.get_weir_outflow(self.reservoir_id)
        self.swat.write_outflow(self.reservoir_id, np.full(len(outflow_df), outflow))

        # run swat
        self.swat.change_start_day(day - self.lookback)
        self.swat.change_end_day(day + self.delta)
        self.swat()

        # read new chl-a concentration
        rch_out = self.swat.read_rch(self.downstream_rch_id, self.year)
        wql_out = self.swat.read_wql_output(self.downstream_rch_id)
        #a = jday_to_date(year, day)

        return wql_out.loc[:, constituent].mean(), rch_out.loc[:, "FLOW_INcms"].mean()

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

    def save_results(self):

        pd.DataFrame(
            np.concatenate([
                np.array(self.step_actions_total).reshape(-1, 1),
                np.array(self.step_rewards_total).reshape(-1, 1),
                np.array(self.step_chla_total).reshape(-1, 1),
                np.array(self.step_states_total).reshape(-1, 1),
                 ], axis=1
            ),
            columns=["actions", "rewards", "chla", "state"]
        ).to_csv(os.path.join(self.path, "results.csv"))

        pd.DataFrame(
            np.concatenate([
                np.array(self.step_actions).reshape(-1,1),
                np.array(self.step_rewards).reshape(-1, 1),
                np.array(self.step_states).reshape(-1, 1),
                np.array(self.step_chla).reshape(-1, 1),
            ], axis=1),
            columns=["actions", "rewards", "chla", "state"]
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

        return

