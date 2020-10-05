
LSTM_CONFIG = {'nn_0': {'lstm_0': {'config':  {'units': 64,  'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences':False, 'return_state': False, 'name': 'lstm_0'}},
                        'Dense': {'config':  {'units': 1}}
                        },

               'nn_1': {'lstm_0': {'config':  {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                          'return_sequences':False, 'return_state': False, 'name': 'lstm_1'}},
                        'Dense': {'config':  {'units': 1}}
                        },

               'nn_2': {'lstm_0': {'config':  {'units': 16, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_2'}},
                        'Dense': {'config':  {'units': 1}}
                        },

               'nn_3': {'lstm_0': {'config':  {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_3'}},
                        'Dense': {'config':  {'units': 1}}
                        }
               }


TCN_CONFIG = {'nn_0': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense': {'config':  {'units': 1}}
                       },

               'nn_1': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense': {'config':  {'units': 1}}
                        },

               'nn_2': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense': {'config':  {'units': 1}}
                        },

               'nn_3': {"tcn": {'config':  {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0}},
                       'Dense': {'config':  {'units': 1}}
                        }
               }


ConvLSTM_CONFIG = {'nn_0': {'convlstm2d': {'config':  {'filters': int(64), 'kernel_size': (1,int(2)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten': {'config':  {}},
                            'relu' : {'config':  {}},
                            'Dense': {'config':  {'units': 1}}
                            },
                   'nn_1': {'convlstm2d': {'config':  {'filters': int(64), 'kernel_size': (1,int(3)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten': {'config':  {}},
                            'leakyrelu' : {'config':  {}},
                            'Dense': {'config':  {'units': 1}}
                            },
                   'nn_2': {'convlstm2d': {'config':  {'filters': int(64), 'kernel_size': (1,int(4)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten': {'config':  {}},
                            'elu' : {'config':  {}},
                            'Dense': {'config':  {'units': 1}}
                            },
                   'nn_3': {'convlstm2d': {'config':  {'filters': int(128), 'kernel_size': (1,int(5)), 'activation': 'relu', 'padding': 'same'}},
                            'flatten': {'config':  {}},
                            'tanh': {'config':  {}},
                            'Dense': {'config':  {'units': 1}}
                            }
                   }


LSTMAutoEnc_Config = {'nn_0': {'lstm_0': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector': {'config':  {'n': 11}},
                               'lstm_1': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "relu_1": {'config':  {}},
                               'Dense': {'config':  {'units': 1}}
                               },
                      'nn_1': {'lstm_0': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector': {'config':  {'n': 11}},
                               'lstm_1': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_1": {'config':  {}},
                               'Dense': {'config':  {'units': 1}}
                               },
                      'nn_2': {'lstm_0': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector': {'config':  {'n': 11}},
                               'lstm_1': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "elu_1": {'config':  {}},
                               'Dense': {'config':  {'units': 1}}
                               },
                      'nn_3': {'lstm_0': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "leakyrelu_0": {'config':  {}},
                               'RepeatVector': {'config':  {'n': 11}},
                               'lstm_1': {'config':  {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
                               "tanh_1": {'config':  {}},
                               'Dense': {'config':  {'units': 1}}
                               }
                      }