# this file shows how to build a simple dense layer based model
# the input_features and outputs are columns and are present in the file

import pandas as pd

from utils import make_model
from models import Model

if __name__ == "__main__":
    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=1,
                                                         inputs=input_features,
                                                         outputs=outputs,
                                                         lr=0.0001)

    df = pd.read_csv('../data/all_data_30min.csv')

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    model.build_nn()

    history = model.train_nn(indices='random')

    y, obs = model.predict(st=0, use_datetime_index=False, marker='.', linestyle='')
    model.view_model(st=0)
