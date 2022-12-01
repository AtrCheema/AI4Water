"""
This sub-module cnotains environments to be used for RL.
"""

from .swat import SWATMultiReservoir, SWATSingleReservoir, SWAT

try:
    from stable_baselines3 import PPO
except (ImportError, ModuleNotFoundError):
    class PPO: pass


class MyPPO(PPO):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        """
        Even if an error is encountered during ``learn``, the results, report and config
        must be saved.

        """
        if exc_type:
            print(f"{exc_type} occured")

        if hasattr(self, 'save_results'):
            self.env.envs[0].save_results(model=self)
            self.env.envs[0].save_plots()

        return