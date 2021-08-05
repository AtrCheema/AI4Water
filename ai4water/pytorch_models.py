__all__ = ["HARHNModel", "IMVModel"]

import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.HARHN import HARHN
from ai4water.imv_networks import IMVTensorLSTM


class HARHNModel(Model):


    def initialize_layers(self, layers_config:dict, inputs=None):
        self.torch_learner.use_cuda = True
        self.pt_model = HARHN(layers_config['n_conv_lyrs'],
                              self.lookback,
                              self.num_ins,
                              self.num_outs,
                              n_units_enc=layers_config['enc_units'],
                              n_units_dec=layers_config['dec_units'],
                              use_predicted_output=True, #self.config['use_predicted_output']
                              ).cuda()

        return

    def forward(self, *inputs: Any, **kwargs: Any):
        y_pred, batch_y_h = self.pt_model(inputs[0], inputs[1][:, -1], **kwargs)
        return y_pred


class IMVModel(Model):

    def initialize_layers(self, layers_config:dict, inputs=None):
        self.torch_learner.use_cuda = True
        self.pt_model = IMVTensorLSTM(self.num_ins, self.num_outs, layers_config['hidden_units']).cuda()
        self.alphas, self.betas = [], []

        return

    def forward(self, *inputs: Any, **kwargs: Any):
        y_pred, alphas, betas = self.pt_model(*inputs, **kwargs)
        self.alphas.append(alphas)
        self.betas.append(betas)
        return y_pred

    def interpret(self, data='training', save=True, **kwargs):
        self.alphas, self.betas = [], []
        self.predict(data=data)


        betas = [array.detach().cpu().numpy() for array in self.betas]
        betas = np.concatenate(betas)  # (samples, ins, 1)
        betas = betas.mean(axis=0)  # (ins, 1)
        betas = betas[..., 0]  # (ins, )

        alphas = [array.detach().cpu().numpy() for array in self.alphas]
        alphas = np.concatenate(alphas)  # (samples, lookback, ins, 1)
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
        ax.set_xticklabels(["t-"+str(i) for i in np.arange(self.lookback, -1, -1)])
        ax.set_yticklabels(list(all_cols))
        for i in range(len(all_cols)):
            for j in range(self.lookback):
                _ = ax.text(j, i, round(alphas[i, j], 3),
                            ha="center", va="center", color="w")
        ax.set_title("Importance of features and timesteps")
        plt.savefig(os.path.join(self.act_path, 'acts'), dpi=400, bbox_inches='tight')

        plt.close('all')
        plt.bar(range(self.num_ins), betas, **kwargs)
        plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=12)
        plt.savefig(os.path.join(self.act_path, 'feature_importance'), dpi=400, bbox_inches='tight')
        return
