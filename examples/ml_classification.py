# How to use AI4Water for classification problems

from ai4water import Model
from ai4water.datasets import MtropicsLaos

data = MtropicsLaos().make_classification()

model = Model(
    data=data,
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1],
    val_fraction=0.0,
    model={"DecisionTreeClassifier": {"max_depth": 4, "random_state": 313}},
    transformation=None,
    problem="classification"
)

h = model.fit()

# make prediction on test data
t, p = model.predict()
#
# # get some useful plots
model.interpret()
#
# # **********Evaluate the model on test data using only input
x, y = model.test_data()
pred = model.evaluate(x=x)  # using only `x`
