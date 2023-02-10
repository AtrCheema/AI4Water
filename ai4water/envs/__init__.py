"""
This sub-module cnotains environments to be used for RL.
"""

from .swat import SWATMultiReservoir, SWATSingleReservoir, SWAT

try:
    from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3, DQN
except (ImportError, ModuleNotFoundError):
    class PPO: pass
    class SAC: pass
    class A2C: pass
    class DDPG: pass
    class TD3: pass
    class DQN: pass


class EnvWithContext(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        """
        Even if an error is encountered during ``learn``, the results, report and config
        must be saved.

        """
        if exc_type:
            print(f"{exc_type} occured")

        if hasattr(self.env.envs[0], 'save_results'):
            self.env.envs[0].save_results(model=self)
            self.env.envs[0].save_plots()

        return


class MyPPO(PPO, EnvWithContext): pass
class MySAC(SAC, EnvWithContext): pass
class MyA2C(A2C, EnvWithContext): pass
class MyDDPG(DDPG, EnvWithContext): pass
class MyTD3(TD3, EnvWithContext): pass
class MyDQN(DQN, EnvWithContext): pass