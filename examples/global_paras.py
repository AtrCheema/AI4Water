
LSTM_CONFIG = {'nn_0': {'lstm_0': {'config':  {'units': 64,  'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences':False, 'return_state': False, 'name': 'lstm_0'}},
                        'Dense_1': {'config':  {'units': 1}}
                        },

               'nn_1': {'lstm_0': {'config':  {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                          'return_sequences':False, 'return_state': False, 'name': 'lstm_1'}},
                        'Dense_2': {'config':  {'units': 1}}
                        },

               'nn_2': {'lstm_0': {'config':  {'units': 16, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_2'}},
                        'Dense_3': {'config':  {'units': 1}}
                        },

               'nn_3': {'lstm_0': {'config':  {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_3'}},
                        'Dense_4': {'config':  {'units': 1}}
                        }
               }


TCN_CONFIG = {'nn_0': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense_1': {'config':  {'units': 1}}
                       },

               'nn_1': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense_2': {'config':  {'units': 1}}
                        },

               'nn_2': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense_3': {'config':  {'units': 1}}
                        },

               'nn_3': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense_4': {'config':  {'units': 1}}
                        }
               }


ConvLSTM_CONFIG = {'nn_0': {'convlstm2d_1': {'config':  {'filters': int(64), 'kernel_size': (1,int(2)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten_1': {'config':  {}},
                            'relu' : {'config':  {}},
                            'Dense_1': {'config':  {'units': 1}}
                            },
                   'nn_1': {'convlstm2d_2': {'config':  {'filters': int(64), 'kernel_size': (1,int(3)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten_2': {'config':  {}},
                            'leakyrelu' : {'config':  {}},
                            'Dense_2': {'config':  {'units': 1}}
                            },
                   'nn_2': {'convlstm2d_3': {'config':  {'filters': int(64), 'kernel_size': (1,int(4)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten_3': {'config':  {}},
                            'elu' : {'config':  {}},
                            'Dense_3': {'config':  {'units': 1}}
                            },
                   'nn_3': {'convlstm2d_3': {'config':  {'filters': int(128), 'kernel_size': (1,int(5)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten_4': {'config':  {}},
                            'tanh': {'config':  {}},
                            'Dense_4': {'config':  {'units': 1}}
                            }
                   }


LSTMAutoEnc_Config = {'nn_0': {'lstm_10': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector_1': {'config':  {'n': 11}},
                               'lstm_11': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "relu_1": {'config':  {}},
                               'Dense_1': {'config':  {'units': 1}}
                               },
                      'nn_1': {'lstm_20': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector_2': {'config':  {'n': 11}},
                               'lstm_21': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_1": {'config':  {}},
                               'Dense_2': {'config':  {'units': 1}}
                               },
                      'nn_2': {'lstm_30': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector_3': {'config':  {'n': 11}},
                               'lstm_31': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "elu_1": {'config':  {}},
                               'Dense_3': {'config':  {'units': 1}}
                               },
                      'nn_3': {'lstm_40': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector_4': {'config':  {'n': 11}},
                               'lstm_41': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "tanh_1": {'config':  {}},
                               'Dense_4': {'config':  {'units': 1}}
                               }
                      }