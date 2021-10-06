__all__ = ["HARHNModel", "IMVModel"]

import os
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.HARHN import HARHN
from ai4water.backend import torch
from ai4water.imv_networks import IMVTensorLSTM
from ai4water.utils.utils import dateandtime_now, plot_activations_along_inputs


class HARHNModel(Model):

    def __init__(self, use_gpu=True, **kwargs):

        dev = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            dev = torch.device("cuda")

        self.dev = dev

        super(HARHNModel, self).__init__(**kwargs)

        # should be set after initiating upper classes so that torch_learner attribute is set
        self.torch_learner.use_cuda = use_gpu

    def initialize_layers(self, layers_config:dict, inputs=None):

        self.pt_model = HARHN(layers_config['n_conv_lyrs'],
                              self.lookback,
                              self.num_ins,
                              self.num_outs,
                              n_units_enc=layers_config['enc_units'],
                              n_units_dec=layers_config['dec_units'],
                              use_predicted_output=True, #self.config['use_predicted_output']
                              ).to(self.dev)

        return

    def forward(self, *inputs: Any, **kwargs: Any):
        y_pred, _ = self.pt_model(inputs[0], inputs[1][:, -1], **kwargs)
        return y_pred


class IMVModel(HARHNModel):

    def initialize_layers(self, layers_config:dict, inputs=None):

        self.pt_model = IMVTensorLSTM(self.num_ins, self.num_outs,
                                      layers_config['hidden_units'],
                                      device=self.dev).to(self.dev)
        self.alphas, self.betas = [], []

        return

    def forward(self, *inputs: Any, **kwargs: Any):
        y_pred, alphas, betas = self.pt_model(*inputs, **kwargs)
        self.alphas.append(alphas)
        self.betas.append(betas)
        return y_pred

    def interpret(self,
                  data='training',
                  x=None,
                  annotate=True,
                  vmin=None,
                  vmax=None,
                  **bar_kws,
                  ):

        mpl.rcParams.update(mpl.rcParamsDefault)

        self.alphas, self.betas = [], []
        true, predicted = self.predict(data=data, process_results=False, return_true=True)

        name = f'data_on_{dateandtime_now()}' if x is not None else data

        betas = [array.detach().cpu().numpy() for array in self.betas]
        betas = np.concatenate(betas)  # (examples, ins, 1)
        betas = betas.mean(axis=0)  # (ins, 1)
        betas = betas[..., 0]  # (ins, )

        alphas = [array.detach().cpu().numpy() for array in self.alphas]
        alphas = np.concatenate(alphas)  # (examples, lookback, ins, 1)

        x, _ = getattr(self, f'{data}_data')()

        plot_activations_along_inputs(data=x[:, -1, :],  # todo, is -1 correct?
                                      activations=alphas.reshape(-1, self.lookback, self.num_ins),
                                      observations=true,
                                      predictions=predicted,
                                      in_cols=self.in_cols,
                                      out_cols=self.out_cols,
                                      lookback=self.lookback,
                                      name=name,
                                      path=self.act_path,
                                      vmin=vmin,
                                      vmax=vmax
                                      )

        alphas = alphas.mean(axis=0)   # (lookback, ins, 1)
        alphas = alphas[..., 0]  # (lookback, ins)
        alphas = alphas.transpose(1, 0)  # (ins, lookback)

        all_cols = self.in_cols
        plt.close('all')
        fig, ax = plt.subplots()
        fig.set_figwidth(16)
        fig.set_figheight(16)
        _ = ax.imshow(alphas)
        ax.set_xticks(np.arange(self.lookback))
        ax.set_yticks(np.arange(len(all_cols)))
        ax.set_xticklabels(["t-"+str(i) for i in np.arange(self.lookback, 0, -1)])
        ax.set_yticklabels(list(all_cols))
        if annotate:
            for i in range(len(all_cols)):
                for j in range(self.lookback):
                    _ = ax.text(j, i, round(alphas[i, j], 3),
                                ha="center", va="center", color="w")
        ax.set_title("Importance of features and timesteps")
        plt.savefig(os.path.join(self.act_path, f'acts_{name}'), dpi=400, bbox_inches='tight')

        plt.close('all')
        plt.bar(range(self.num_ins), betas, **bar_kws)
        plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=12)
        plt.savefig(os.path.join(self.act_path, f'feature_importance_{name}'), dpi=400, bbox_inches='tight')
        return
