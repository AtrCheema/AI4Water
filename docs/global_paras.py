
LSTM_CONFIG = {'nn_0': {'lstm_0': {'units': 64,  'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences':False, 'return_state': False, 'name': 'lstm_0'},
                        'Dense': {'units': 1}
                        },

               'nn_1': {'lstm_0': {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                          'return_sequences':False, 'return_state': False, 'name': 'lstm_1'},
                        'Dense': {'units': 1}
                        },

               'nn_2': {'lstm_0': {'units': 16, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_2'},
                        'Dense': {'units': 1}
                        },

               'nn_3': {'lstm_0': {'units': 256, 'activation': 'leakyrelu', 'dropout': 0.4, 'recurrent_dropout': 0.4,
                                   'return_sequences': False, 'return_state': False, 'name': 'lstm_3'},
                        'Dense': {'units': 1}
                        }
               }


TCN_CONFIG = {'nn_0': {"tcn": {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0},
                       'Dense': {'units': 1}
                       },

               'nn_1': {"tcn": {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0},
                       'Dense': {'units': 1}
                        },

               'nn_2': {"tcn": {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0},
                       'Dense': {'units': 1}
                        },

               'nn_3': {"tcn": {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal', 'use_skip_connections': True, 'return_sequences': False, 'dropout_rate': 0.0},
                       'Dense': {'units': 1}
                        }
               }


ConvLSTM_CONFIG = {'nn_0': {'convlstm2d': {'filters': int(64), 'kernel_size': (1,int(2)), 'activation': 'relu', 'padding': 'same'},
                            'flatten': {},
                            'relu' : {},
                            'Dense': {'units': 1}
                            },
                   'nn_1': {'convlstm2d': {'filters': int(64), 'kernel_size': (1,int(3)), 'activation': 'relu', 'padding': 'same'},
                            'flatten': {},
                            'leakyrelu' : {},
                            'Dense': {'units': 1}
                            },
                   'nn_2': {'convlstm2d': {'filters': int(64), 'kernel_size': (1,int(4)), 'activation': 'relu', 'padding': 'same'},
                            'flatten': {},
                            'elu' : {},
                            'Dense': {'units': 1}
                            },
                   'nn_3': {'convlstm2d': {'filters': int(128), 'kernel_size': (1,int(5)), 'activation': 'relu', 'padding': 'same'},
                            'flatten': {},
                            'tanh': {},
                            'Dense': {'units': 1}
                            }
                   }


LSTMAutoEnc_Config = {'nn_0': {'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "leakyrelu_0": {},
                               'RepeatVector': {'n': 11},
                               'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "relu_1": {},
                               'Dense': {'units': 1}
                               },
                      'nn_1': {'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "leakyrelu_0": {},
                               'RepeatVector': {'n': 11},
                               'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "leakyrelu_1": {},
                               'Dense': {'units': 1}
                               },
                      'nn_2': {'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "leakyrelu_0": {},
                               'RepeatVector': {'n': 11},
                               'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "elu_1": {},
                               'Dense': {'units': 1}
                               },
                      'nn_3': {'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "leakyrelu_0": {},
                               'RepeatVector': {'n': 11},
                               'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               "tanh_1": {},
                               'Dense': {'units': 1}
                               }
                      }