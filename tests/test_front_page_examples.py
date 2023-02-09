import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import tensorflow as tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model


class TestFrontPage(unittest.TestCase):

    def test_example1(self):
        from ai4water import Model
        from ai4water.models import MLP
        from ai4water.datasets import mg_photodegradation
        data, *_ = mg_photodegradation(encoding="le")

        model = Model(
            # define the model/algorithm
            model=MLP(units=24, activation="relu", dropout=0.2),
            # columns in data file to be used as input
            input_features=data.columns.tolist()[0:-1],
            # columns in csv file to be used as output
            output_features=data.columns.tolist()[-1:],
            lr=0.001,  # learning rate
            batch_size=8,  # batch size
            epochs=500,  # number of epochs to train the neural network
            patience=50,  # used for early stopping
        )

        history = model.fit(data=data)

        prediction = model.predict_on_test_data(data=data)

        prediction = model.predict_on_test_data(data=data)

        import tensorflow as tf
        assert isinstance(model, tf.keras.Model)  # True
        return

    def test_example2(self):
        from ai4water.models import LSTM
        batch_size = 16
        lookback = 15
        inputs = ['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5']  # just dummy names for plotting and saving results.
        outputs = ['DummyTarget']

        model = Model(
            model=LSTM(units=64),
            batch_size=batch_size,
            ts_args={'lookback': lookback},
            input_features=inputs,
            output_features=outputs,
            lr=0.001
        )
        x = np.random.random((batch_size * 10, lookback, len(inputs)))
        y = np.random.random((batch_size * 10, len(outputs)))

        model.fit(x=x, y=y)
        return

    def test_example3(self):
        from ai4water import Model
        from ai4water.datasets import busan_beach

        data = busan_beach()  # path for data file

        model = Model(
            # columns in data to be used as input
            input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'rel_hum', 'pcp_mm'],
            output_features=['tetx_coppml'],  # columns in data file to be used as input
            seed=1872,
            val_fraction=0.0,
            split_random=True,
            #  any regressor from https://scikit-learn.org/stable/modules/classes.html
            model={"RandomForestRegressor": {}},
            # set any of regressor's parameters. e.g. for RandomForestRegressor above used,
            # some of the paramters are https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        )

        history = model.fit(data=data)

        model.predict_on_test_data(data=data)
        return

    def test_example4(self):
        from ai4water.functional import Model
        from ai4water.datasets import MtropicsLaos
        from ai4water.hyperopt import Real, Integer

        data = MtropicsLaos().make_regression(lookback_steps=1)

        model = Model(
            model={"RandomForestRegressor": {
                "n_estimators": Integer(low=5, high=30, name='n_estimators', num_samples=10),
                "max_leaf_nodes": Integer(low=2, high=30, prior='log', name='max_leaf_nodes', num_samples=10),
                "min_weight_fraction_leaf": Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=10),
                "max_depth": Integer(low=2, high=10, name='max_depth', num_samples=10),
                "min_samples_split": Integer(low=2, high=10, name='min_samples_split', num_samples=10),
                "min_samples_leaf": Integer(low=1, high=5, name='min_samples_leaf', num_samples=10),
            }},
            input_features=data.columns.tolist()[0:-1],
            output_features=data.columns.tolist()[-1:],
            cross_validator={"KFold": {"n_splits": 5}},
            x_transformation="zscore",
            y_transformation="log",
        )

        # First check the performance on test data with default parameters
        model.fit_on_all_training_data(data=data)
        print(model.evaluate_on_test_data(data=data, metrics=["r2_score", "r2"]))

        # optimize the hyperparameters
        optimizer = model.optimize_hyperparameters(
            algorithm="bayes",  # you can choose between `random`, `grid` or `tpe`
            data=data,
            num_iterations=20,  # todo
        )

        # Now check the performance on test data with default parameters
        print(model.evaluate_on_test_data(data=data, metrics=["r2_score", "r2"]))
        return

    def test_example5(self):
        from ai4water.datasets import busan_beach
        from ai4water.experiments import MLRegressionExperiments

        data = busan_beach()

        comparisons = MLRegressionExperiments(
            input_features=data.columns.tolist()[0:-1],
            output_features=data.columns.tolist()[-1:],
            split_random=True
        )
        # train all the available machine learning models
        comparisons.fit(data=data)
        # Compare R2 of models
        best_models = comparisons.compare_errors(
            'r2',
            data=data,
            cutoff_type='greater',
            cutoff_val=0.1,
            figsize=(8, 9),
            colors=['salmon', 'cadetblue']
        )
        # Compare model performance using Taylor diagram
        _ = comparisons.taylor_plot(
            data=data,
            figsize=(5, 9),
            exclude=["DummyRegressor", "XGBRFRegressor",
                     "SGDRegressor", "KernelRidge", "PoissonRegressor"],
            leg_kws={'facecolor': 'white',
                     'edgecolor': 'black', 'bbox_to_anchor': (2.0, 0.9),
                     'fontsize': 10, 'labelspacing': 1.0, 'ncol': 2
                     },
        )
        return


if __name__ == "__main__":

    unittest.main()
