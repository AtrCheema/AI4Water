
import os

import numpy as np
import pandas as pd

from ai4water.backend import wandb
from .utils.utils import get_version_info
from .utils import jsonize


class WB:

    def __init__(
            self,
            model,
            config:dict
    ):
        self.wandb_config = config.copy()
        self.model = model
        self.init()

    def __getattr__(self, item):
        return getattr(self.model, item)

    def init(self):
        """calls wandb.init and creates wb_run_ attribute"""

        assert wandb is not None, """wandb is not installed. 
        Please install it using pip install wandb"""

        target = getattr(self, 'output_features', None)
        category = getattr(self, 'category', None)
        mode = getattr(self, 'mode', None)
        path = getattr(self, 'path', None)
        model_name = getattr(self, 'model_name', None)
        val_metric = getattr(self, 'val_metric', None)
        input_features = getattr(self, 'input_features', None)
        config = getattr(self, 'config', {})
        name = None

        if isinstance(target, list):
            target = target[0]

        if target is not None and path:
            name = f"{target[0:7]}_{mode}_{category}_{os.path.basename(path)[-15:]}"
        elif path:
            name = f"{mode}_{category}_{os.path.basename(self.path)[-15:]}"

        def_tags = [category, mode,
                    f"{model_name}",
                    val_metric]

        if input_features is not None:
            def_tags += [f"{len(input_features)}_inputs"]

        def_tags += [f"target_{target}"]

        init_config = dict(
            config=jsonize(config),
            notes=f"{mode} with {category}",
            tags=def_tags,
            name=name
        )

        wandb_config = self.wandb_config.copy()

        for k in ['training_data', 'validation_data', 'monitor']:
            wandb_config.pop(k, None)

        init_config.update(wandb_config)

        run = wandb.init(**init_config)

        setattr(self, 'wb_run_', run)
        return

    def callbacks_tf(self, callbacks, train_data, validation_data):
        """makes callbacks for WB"""
        from wandb.keras import WandbCallback

        if callbacks is None:
            callbacks = {}

        wandb_config: dict = self.config['wandb_config']

        monitor = wandb_config.get('monitor', None)

        kws = {}
        if monitor == "val_loss" and validation_data is not None:
            kws['monitor'] = monitor
            kws['validation_data'] = validation_data
            kws['save_model'] = False

            callbacks['wandb_callback'] = WandbCallback(**kws)

        return callbacks

    def log_predict(self, true, prediction, mode, prefix=''):
        """logs prediction results on wb. """
        if mode == "regression" and true is not None:

            data = pd.DataFrame(
                np.column_stack([true, prediction]),
                columns=['true', 'prediction'])

            table = wandb.Table(data=data, columns=["true", "prediction"])
            self.wb_run_.log({f"scatter_{prefix}": wandb.plot.scatter(
                table,
                "true", "prediction")})
        return

    def log_loss_curve(self, history, prefix=''):
        """plots the loss curve on wb."""
        if history is not None:
            self.wb_run_.log({"loss": history.history['loss']})
            if 'val_loss' in history.history:
                self.wb_run_.log({f"val_loss_{prefix}": history.history['val_loss']})
        return
    
    def on_epoch_end(self, epoch, train_losses, val_losses=None, prefix=''):
        """logs epoch end results on wb."""
        self.wb_run_.log(train_losses, step=epoch)
        if val_losses:
            self.wb_run_.log(val_losses, step=epoch)
        return

    def finish(self):
        """does some stuff related to wandb at the end of training."""

        notes = 'Following are versions of libraries used. \n'

        for lib, ver in get_version_info().items():
            notes += f"{lib}: {ver}\n"

        self.wb_run_.notes = notes

        self.wb_run_.finish()
        return