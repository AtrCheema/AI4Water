

from ai4water.experiments import MLClassificationExperiments

from ai4water.datasets import MtropicsLaos

data = MtropicsLaos().make_classification(lookback_steps=2)

inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

exp = MLClassificationExperiments(input_features=inputs,
                                  output_features=outputs)

exp.fit(data=data)
exp.compare_errors('accuracy')