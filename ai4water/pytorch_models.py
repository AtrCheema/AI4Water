__all__ = ["HARHNModel", "IMVModel"]

import os
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import dateandtime_now, plot_activations_along_inputs
from ai4water.backend import torch

if torch is not None:
    from ai4water.models.torch import IMVTensorLSTM
    from ai4water.models.torch import HARHN
else:
    HARHN, IMVTensorLSTM = None, None

from ai4water.utils.easy_mpl import imshow


class HARHNModel(Model):

    def __init__(self, use_gpu=True, teacher_forcing=True, **kwargs):

        dev = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            dev = torch.device("cuda")

        self.dev = dev

        super(HARHNModel, self).__init__(teacher_forcing=teacher_forcing, **kwargs)

        # should be set after initiating upper classes so that torch_learner attribute is set
        self.torch_learner.use_cuda = use_gpu

    def initialize_layers(self, layers_config: dict, inputs=None):

        self.pt_model = HARHN(layers_config['n_conv_lyrs'],
                              self.lookback,
                              self.num_ins,
                              self.num_outs,
                              n_units_enc=layers_config['enc_units'],
                              n_units_dec=layers_config['dec_units'],
                              use_predicted_output=self.teacher_forcing,  # self.config['use_predicted_output']
                              ).to(self.dev)

        return

    def forward(self, *inputs: Any, **kwargs: Any):
        y_pred, _ = self.pt_model(inputs[0], inputs[1][:, -1], **kwargs)
        return y_pred


class IMVModel(HARHNModel):

    def __init__(self, *args, teacher_forcing=False, **kwargs):
        super(IMVModel, self).__init__(*args, teacher_forcing=teacher_forcing, **kwargs)

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

        path = os.path.join(self.path, "interpret")
        if not os.path.exists(path):
            os.makedirs(path)

        plot_activations_along_inputs(data=x[:, -1, :],  # todo, is -1 correct?
                                      activations=alphas.reshape(-1, self.lookback, self.num_ins),
                                      observations=true,
                                      predictions=predicted,
                                      in_cols=self.input_features,
                                      out_cols=self.output_features,
                                      lookback=self.lookback,
                                      name=name,
                                      path=path,
                                      vmin=vmin,
                                      vmax=vmax
                                      )

        alphas = alphas.mean(axis=0)   # (lookback, ins, 1)
        alphas = alphas[..., 0]  # (lookback, ins)
        alphas = alphas.transpose(1, 0)  # (ins, lookback)

        all_cols = self.input_features
        plt.close('all')
        fig, ax = plt.subplots()
        fig.set_figwidth(16)
        fig.set_figheight(16)
        xticklabels=["t-"+str(i) for i in np.arange(self.lookback, 0, -1)]
        imshow(alphas,
               ax=ax,
               xticklabels=xticklabels,
               yticklabels=list(all_cols),
               title="Importance of features and timesteps",
               annotate=annotate,
               show=False)


        plt.savefig(os.path.join(path, f'acts_{name}'), dpi=400, bbox_inches='tight')

        plt.close('all')
        plt.bar(range(self.num_ins), betas, **bar_kws)
        plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=12)
        plt.savefig(os.path.join(path, f'feature_importance_{name}'), dpi=400, bbox_inches='tight')
        return
