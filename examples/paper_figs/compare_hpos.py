# This file shows comparison of different hyper-parameter optimization algorithms.
import os
import json
import random
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from AI4Water.utils.datasets import CAMELS_AUS
from AI4Water import Model
from AI4Water.hyper_opt import HyperOpt, Categorical, Real, Integer
from AI4Water.utils.utils import dateandtime_now

tf.compat.v1.disable_eager_execution()

seed = 313
np.random.seed(seed)
random.seed(seed)

dataset = CAMELS_AUS(path=r"D:\mytools\AI4Water\AI4Water\utils\datasets\CAMELS\CAMELS_AUS")

inputs = ['et_morton_point_SILO',
           'precipitation_AWAP',
           'tmax_AWAP',
           'tmin_AWAP',
           'vprp_AWAP',
           'rh_tmax_SILO',
           'rh_tmin_SILO']

outputs = ['streamflow_MLd']

data = dataset.fetch(['224206'], dynamic_attributes=inputs+outputs, categories=None, st='19700101', en='20141231')

for k,v in data.items():
    target = v['streamflow_MLd']
    target[target < 0] = 0
    v['streamflow_MLd'] = target
    data[k] = v
    print(k, v.isna().sum().sum())

results = {}
suffix = f'hpo_comparisons_{dateandtime_now()}'
for m in ['tpe',
          'bayes',
          'grid',
          'random'
            ]:

    _suffix = f"{suffix}\\{m}"

    def objective_fn(**suggestion):

        print(suggestion, 'suggestion')

        model = Model(
            model={'layers': {'lstm': {'config': {'units': 64,
                                                  'activation': suggestion['activation'],
                                                  'dropout': 0.2,
                                                  'recurrent_dropout': 0.2}}}},
            inputs=inputs,
            outputs=outputs,
            lookback=int(suggestion['lookback']),
            lr=float(suggestion['lr']),
            batch_size=int(suggestion['batch_size']),
            data=data['224206'],
            verbosity=0,
            epochs=500,
            prefix=_suffix
        )

        h = model.fit()
        return np.min(h.history['val_loss'])

    num_samples=4
    d = [
        Categorical(categories=['relu', 'sigmoid', 'tanh', 'linear'], name='activation'),
        Integer(low=3, high=15, name='lookback', num_samples=num_samples),
        Categorical(categories=[16, 32, 64, 128], name='batch_size'),
        Real(low=1e-5, high=0.001, name='lr', num_samples=num_samples)
    ]
    x0 = ['relu', 5, 32, 0.0001]

    optimizer = HyperOpt(m, objective_fn=objective_fn, param_space=d, num_iterations=50, x0=x0,
                         use_named_args=True,
                         opt_path=os.path.join(os.getcwd(), f'results\\{_suffix}'))

    r = optimizer.fit()
    results[m] = optimizer


with open(r'D:\mytools\AI4Water\examples\results\hpo_comparisons_20210310_153743\grid\eval_results.json', 'r') as fp:
    eval_results = json.load(fp)
results = {int(key.split('_')[1]): float(key.split('_')[0]) for key in eval_results.keys()}
results = OrderedDict(results)
results = dict(sorted(results.items()))
results = np.array(list(results.values()))[0:144]
iterations = range(1, len(results) + 1)
grid = [np.min(results[:i]) for i in iterations]

with open(r'D:\mytools\AI4Water\examples\results\hpo_comparisons_20210310_153743\random\eval_results_random.json', 'r') as fp:
    eval_results_random = json.load(fp)

results = {int(key.split('_')[1]): float(key.split('_')[0]) for key in eval_results_random.keys()}
results = OrderedDict(results)
results = dict(sorted(results.items()))
results = np.array(list(results.values()))[0:144]
iterations = range(1, len(results) + 1)
random = [np.min(results[:i]) for i in iterations]

with open(r'D:\mytools\AI4Water\examples\results\hpo_comparisons_20210310_153743\tpe\trials.json', 'r') as fp:
    trials = json.load(fp)
tpe = [trials[i]['result']['loss'] for i in range(len(trials))]
iterations = range(1, len(tpe) + 1)
tpe = [np.min(tpe[:i]) for i in iterations]

with open(r'D:\mytools\AI4Water\examples\results\hpo_comparisons_20210310_153743\bayes\gp_parameters.json', 'r') as fp:
    gp_parameters = json.load(fp)
bayes = gp_parameters['func_vals']
iterations = range(1, len(bayes) + 1)
bayes = [np.min(bayes[:i]) for i in iterations]


plt.close('all')
fig, axis = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(9)
axis.grid()
axis.plot(tpe, '--s', label='TPE')
axis.plot(bayes, '--*', label='GP')
axis.plot(grid, '--.', label='Grid')
axis.plot(random, '--^', label='Random')

axis.legend(fontsize=20, markerscale=2)
axis.set_xlabel('Number of iterations $n$', fontsize=20)
axis.set_ylabel(r"$\min f(x)$ after $n$ calls", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()