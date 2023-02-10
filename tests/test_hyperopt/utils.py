
from ai4water import Model
from ai4water.backend import os, np, skopt, optuna
from ai4water.utils.utils import dateandtime_now
from ai4water.datasets import busan_beach
from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from ai4water.hyperopt import HyperOpt, Real, Categorical, Integer

if skopt is not None:
    from skopt.space.space import Space

data = busan_beach()
inputs = list(data.columns)[0:-1]
outputs = [list(data.columns)[-1]]


def check_attrs(optimizer, paras):
    optimizer.eval_with_best()
    space = optimizer.space()
    assert isinstance(space, dict)
    assert len(space)==paras
    assert isinstance(optimizer.xy_of_iterations(), dict)
    assert len(optimizer.xy_of_iterations()) >= optimizer.num_iterations, f"""
            {len(optimizer.xy_of_iterations())}, {optimizer.num_iterations}"""
    assert isinstance(optimizer.best_paras(as_list=True), list)
    assert isinstance(optimizer.best_paras(False), dict)
    assert len(optimizer.func_vals()) >= optimizer.num_iterations
    assert isinstance(optimizer.func_vals(), np.ndarray)

    # all keys should be integer but we check only first
    first_key = list(optimizer.xy_of_iterations().keys())[0]
    assert isinstance(first_key, int)
    first_xy = list(optimizer.xy_of_iterations().values())[0]
    assert isinstance(first_xy['x'], dict)
    assert isinstance(first_xy['y'], float)

    if skopt is not None:
        assert isinstance(optimizer.skopt_space(), Space)

    fpath = os.path.join(optimizer.opt_path, 'serialized.json')
    assert os.path.exists(fpath)

    return



def run_unified_interface(algorithm,
                          backend,
                          num_iterations,
                          num_samples=None,
                          process_results=True,
                          use_kws=False,
                          ):

    prefix = f'test_{algorithm}_xgboost_{backend}{dateandtime_now()}'

    if use_kws:
        def fn(**suggestion):
            model = Model(
                input_features=inputs,
                output_features=outputs,
                model={"RandomForestRegressor": suggestion},
                prefix=prefix,
                split_random=True,
                verbosity=0)

            model.fit(data=data)

            t, p = model.predict_on_validation_data(data=data, return_true=True, process_results=False)
            mse = RegressionMetrics(t, p).mse()

            return mse
    else:

        def fn(a=2, **suggestion):
            model = Model(
                input_features=inputs,
                output_features=outputs,
                model={"RandomForestRegressor": suggestion},
                prefix=prefix,
                verbosity=0)

            return np.random.randn()

    search_space = [
        Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features'),
        Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
        Real(low=0.1, high=0.5, name='min_samples_split', num_samples=num_samples),
    ]

    optimizer = HyperOpt(algorithm, objective_fn=fn, param_space=search_space,
                         backend=backend,
                         num_iterations=num_iterations,
                         verbosity=0,
                         opt_path=os.path.join(os.getcwd(), f'results\\{prefix}'),
                         process_results=process_results
                         )

    optimizer.fit()
    check_attrs(optimizer, len(search_space))

    files_to_check = ["convergence.png",
                      "iterations.json",
     "edf.png", "parallel_coordinates.png", "iterations_sorted.json",
     'distributions.png',
     ]

    if optuna is not None:
        files_to_check += ["fanova_importance_hist.png", "fanova_importance_bar.png"]

    if process_results:

        for f in files_to_check:
            fpath = os.path.join(optimizer.opt_path, f)
            assert os.path.exists(fpath), fpath
    return optimizer
