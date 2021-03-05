__all__ = ['TemporalFusionTransformer']

# This file contain most of the code from
# https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
# The TemporalFusionTransformer class has been modified so that it can be used as regular layer and without static,
# categorical, observation and future inputs. The original code had Apache Licence, Version 2.0 although a lot of
# the code has been modified here. http://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import TimeDistributed, Dense, InputLayer, Embedding

from dl4seq.models.utils import concatenate, gated_residual_network, add_and_norm, apply_gating_layer, get_decoder_mask
from dl4seq.models.utils import InterpretableMultiHeadAttention


# Layer definitions.
stack = tf.keras.backend.stack


class TemporalFusionTransformer(tf.keras.layers.Layer):
    """
    1, 11, 21, 31, a
    2, 12, 22, 32, b
    3, 13, 23, 33, c
    4, 14, 24, 34, d

    known_categorical_inputs: a,b,c
    obs_inputs
    unknown_inputs
    static_inputs
    time_steps: int, lookback + horizons. Total number of input time steps per forecast date
    input_size: int, number of input features
    _static_input_loc: None/list, location of static inputs
    known_categorical_inputs: list,
    future_inputs=bool, default False, whether we have future data as input or not.
    category_counts: list, Number of categories per categorical variable
    _static_input_loc: list
    use_cudnn: Whether to use Keras CuDNNLSTM or standard LSTM layers
    hidden_units: Internal state size of TFT


    """
    def __init__(self, raw_params:dict, **kwargs):

        self.time_steps = int(raw_params['total_time_steps'])
        self.input_size = int(raw_params['num_inputs'])
        self._known_regular_input_idx = raw_params['known_regular_inputs']  # [1,2,3]
        self._input_obs_loc = raw_params['input_obs_loc'] # [0]
        self._static_input_loc = raw_params['static_input_loc']  #[3,4]
        self.category_counts = raw_params['category_counts'] #[2, 2]
        self._known_categorical_input_idx = raw_params['known_categorical_inputs']

        # Network params
        self.use_cudnn = raw_params['use_cudnn']  # Whether to use GPU optimised LSTM
        self.hidden_units = int(raw_params['hidden_units'])
        self.dropout_rate = float(raw_params['dropout_rate'])
        self.encoder_steps = int(raw_params['num_encoder_steps'])  # historical steps/lookback steps
        self.num_heads = int(raw_params['num_heads'])

        self.future_inputs = raw_params.get('future_inputs', False)
        self.return_attention_components = raw_params.get('return_attention_components', False)

        super().__init__(**kwargs)

    def __call__(self, alle_inputs, *args, **kwargs):
        """Returns graph defining layers of the TFT.
        """

        # Size definitions.
        encoder_steps = self.encoder_steps  # encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(alle_inputs)

        # known_combined_layer.shape = (?, time_steps, hidden_units, num_outputs)
        # obs_inputs.shape = (?, time_steps, hidden_units, 1)
        # static_inputs.shape = (?, num_cat_variables, hidden_units)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concatenate([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], axis=-1, name="historical_inputs")
        else:
            if obs_inputs is not None:
                # we are extracting only historical data i.e. lookback from obs_inputs
                # (?, encoder_steps, hidden_units, 4) <- (?, encoder_steps, hidden_units, num_outputs) | (?, encoder_steps, hidden_units, 1)
                historical_inputs = concatenate([
                    known_combined_layer[:, :encoder_steps, :],
                    obs_inputs[:, :encoder_steps, :]], axis=-1, name="historical_inputs")
            else:
                historical_inputs = known_combined_layer[:, :encoder_steps, :]

        if self.future_inputs:
            assert self.time_steps - self.encoder_steps > 0
            # Isolate only known future inputs.
            future_inputs = known_combined_layer[:, encoder_steps:, :]  # (?, 24, hidden_units, num_outputs)
        else:
            future_inputs=None

        def static_combine_and_mask(embedding):
            """Applies variable selection network to static inputs.

            Args:
              embedding: Transformed static inputs (?, num_cat_variables, 160)

            Returns:
              Tensor output for variable selection network
            """

            # Add temporal features
            _, num_static, _ = embedding.get_shape().as_list()

            flatten = tf.keras.layers.Flatten()(embedding)  # (?, hidden_units*num_cat_variables)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = gated_residual_network(  # (?, num_cat_variables)
                flatten,
                self.hidden_units,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None,
                name='GRN_static'
            )

            # (?, num_cat_variables)
            sparse_weights = tf.keras.layers.Activation('softmax', name='sparse_weights')(mlp_outputs)
            sparse_weights = K.expand_dims(sparse_weights, axis=-1)  # (?, num_cat_variables, 1)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(  # e.shape = (?, 1, hidden_units)
                    embedding[:, i:i + 1, :],
                    self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                    name=f'GRN_static_{i}'
                )
                trans_emb_list.append(e)

            # (?, num_cat_variables, hidden_units)
            transformed_embedding = concatenate(trans_emb_list, axis=1, name="transfomred_embedds")

            combined = tf.keras.layers.Multiply()(  # (?, num_cat_variables, hidden_units)
                [sparse_weights, transformed_embedding])

            static_vec = K.sum(combined, axis=1)  # (?, hidden_units)

            return static_vec, sparse_weights

        static_context_state_h = None
        static_context_state_c = None
        static_context_variable_selection = None
        static_context_enrichment = None
        static_weights = None

        if static_inputs is not None:
            static_encoder, static_weights = static_combine_and_mask(static_inputs)

            # static_encoder.shape = (?, hidden_units)
            # static_weights.shape = (?, num_cat_variables, 1)
            static_context_variable_selection = gated_residual_network(  # (?, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GNR_st_cntxt_var_select"
            )
            static_context_enrichment = gated_residual_network(  # (?, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_enrich"
            )
            static_context_state_h = gated_residual_network(  # (?, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_h"
            )
            static_context_state_c = gated_residual_network(  # (?, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_c"
            )

        def lstm_combine_and_mask(embedding, static_context, _name=None):
            """Apply temporal variable selection networks.

            Args:
              embedding: Transformed inputs. (?, encoder_steps, hidden_units, 4)
              static_context:
              _name: name of encompassing layers

            Returns:
              Processed tensor outputs.
            """

            # Add temporal features
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

            flatten = K.reshape(embedding,  # (?, encoder_steps, 640)
                                [-1, time_steps, embedding_dim * num_inputs])

            if static_context is not None:
                _expanded_static_context = K.expand_dims(  # (?, 1, hidden_units)
                    static_context, axis=1)
            else:
                _expanded_static_context = None

            # Variable selection weights
            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_units,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=_expanded_static_context,
                return_gate=True,
                name=f'GRN_with_{_name}'
            )

            # mlp_outputs.shape (?, encoder_steps, 4)
            # static_gate.shape (?, encoder_steps, 4)
            # sparse_weights (?, 1, 1)
            sparse_weights = tf.keras.layers.Activation('softmax',
                                                        name=f'sparse_{_name}_weights_softmax')(mlp_outputs)  # (?, encoder_steps, 4)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2)  # (?, encoder_steps, 1, 4)

            # Non-linear Processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[Ellipsis, i],
                    self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                    name=f'GRN_with_{_name}_for_{i}'
                )
                trans_emb_list.append(grn_output)

            transformed_embedding = stack(trans_emb_list, axis=-1)  # (?, encoder_steps, hidden_units, 4)

            combined = tf.keras.layers.Multiply(name=f'sparse_and_transform_{_name}')(  # (?, encoder_steps, hidden_units, 4)
                [sparse_weights, transformed_embedding])
            temporal_ctx = K.sum(combined, axis=-1)  # (?, encoder_steps, hidden_units)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_flags, _ = lstm_combine_and_mask(historical_inputs,
                                                                         static_context_variable_selection,
                                                                         _name='history')
        # historical_features.shape = (?, encoder_steps, hidden_units)
        # historical_flags (?, encoder_steps, 1, 4)
        future_features = None
        future_flags = None
        if future_inputs is not None:
            future_features, future_flags, _ = lstm_combine_and_mask(future_inputs,
                                                                     static_context_variable_selection,
                                                                     _name='future')

        # future_features = (?, 24, hidden_units)
        # future_flags = (?, 24, 1, num_outputs)

        # LSTM layer
        def get_lstm(return_state, _name=None):
            """Returns LSTM cell initialized with default parameters."""
            if self.use_cudnn:
                lstm = tf.keras.layers.CuDNNLSTM(
                    self.hidden_units,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                    name=_name
                )
            else:
                lstm = tf.keras.layers.LSTM(
                    self.hidden_units,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                    # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
                    # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    name=_name
                )
            return lstm

        initial_states = [static_context_state_h, static_context_state_c] if static_context_state_h is not None else None
        history_lstm, state_h, state_c = get_lstm(return_state=True, _name='history')(historical_features,
                                                                     initial_state=initial_states)
        # history_lstm = (?, encoder_steps, hidden_units)

        if future_features is not None:
            future_lstm = get_lstm(return_state=False, _name='future')(
                future_features, initial_state=[state_h, state_c])
            # future_lstm = (?, 24, hidden_units)
            lstm_layer = concatenate([history_lstm, future_lstm], axis=1, name='history_plus_future_lstm')  # (?, time_steps, hidden_units)
            # Apply gated skip connection
            input_embeddings = concatenate([historical_features, future_features], axis=1, name="history_plus_future_embeddings"
                                           )  # (?, time_steps, hidden_units)
        else:
            lstm_layer = history_lstm
            input_embeddings = historical_features

        lstm_layer, _ = apply_gating_layer(  # (?, time_steps, hidden_units)
            lstm_layer, self.hidden_units, self.dropout_rate, activation=None, name='GatingOnLSTM')
        # (?, time_steps, hidden_units)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings], name='AfterLSTM')

        # Static enrichment layers
        expanded_static_context = None
        if static_context_enrichment is not None:
            expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)  # (?, 1, hidden_units)

        enriched, _ = gated_residual_network(  # (?, time_steps, hidden_units)
            temporal_feature_layer,
            self.hidden_units,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            additional_context=expanded_static_context,
            return_gate=True,
            name='GRN_temp_feature'
        )

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_units, dropout=self.dropout_rate, name="InterpMultiHeadAtten")

        mask = get_decoder_mask(enriched)
        x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)
        # x =  (?, time_steps, hidden_units)
        x, _ = apply_gating_layer(  # # x =  (?, time_steps, hidden_units)
            x,
            self.hidden_units,
            dropout_rate=self.dropout_rate,
            activation=None,
            name="GatingOnX"
        )
        x = add_and_norm([x, enriched], name="XAndEnriched")  # # x =  (?, time_steps, hidden_units)

        # Nonlinear processing on outputs
        decoder = gated_residual_network(  # # x =  (?, time_steps, hidden_units)
            x,
            self.hidden_units,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="NonLinearityOnOut"
        )

        # Final skip connection
        decoder, _ = apply_gating_layer(decoder, self.hidden_units, activation=None,
                                        name="FinalSkip")  # # x =  (?, time_steps, hidden_units)

        # (?, time_steps, hidden_units)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer], name="DecoderAndTempFeature")

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,  # (4, ?, time_steps, time_steps)
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0] if static_weights is not None else None,  # (?, 1)
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :] if future_flags is not None else None
        }

        self.attention_components = attention_components

        if self.return_attention_components:
            return transformer_layer, attention_components

        return transformer_layer

    def get_tft_embeddings(self, all_inputs):
        """Transforms raw inputs to embeddings.

        Applies linear transformation onto continuous variables and uses embeddings
        for categorical variables.

        Args:
          all_inputs: Inputs to transform [batch_size, time_steps, input_features]
          whre time_steps include both lookback and forecast. The input_features dimention of all_inputs can
          contain following inputs. `static_inputs`, `obs_inputs`, `categorical_inputs`, `regular_inputs`.

        Returns:
          Tensors for transformed inputs.
          unknown_inputs:
          known_combined_layer: contains regular inputs and categorical inputs (all known)
          obs_inputs: target values to be used as inputs.
          static_inputs
        """

        time_steps = self.time_steps

        # Sanity checks
        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError('Observation cannot be known a priori!')
        if self._input_obs_loc is not None:
            for i in self._input_obs_loc:
                if i in self._static_input_loc:
                    raise ValueError('Observation cannot be static!')

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
          raise ValueError(
              'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                  all_inputs.get_shape().as_list()[-1], self.input_size))


        num_categorical_variables = len(self.category_counts)  # 1
        num_regular_variables = self.input_size - num_categorical_variables  # 4

        embeddings = []
        if num_categorical_variables > 0:
            embedding_sizes = [   # [hidden_units]
                self.hidden_units for i, size in enumerate(self.category_counts)
            ]

            for i in range(num_categorical_variables):

                embedding = tf.keras.Sequential([InputLayer([time_steps]),Embedding(
                    self.category_counts[i],
                    embedding_sizes[i],
                    input_length=time_steps,
                    dtype=tf.float32)
                                                 ])
                embeddings.append(embedding)

            regular_inputs, categorical_inputs = all_inputs[:, :, :num_regular_variables], \
                                                 all_inputs[:, :, num_regular_variables:]

            # regular_inputs (?, time_steps, 4)
            # categorical_inputs (?, time_steps, num_static_inputs)
            # list of lengnth=(num_static_inputs) with shape (?, time_steps, hidden_units)
            embedded_inputs = [
                embeddings[i](categorical_inputs[Ellipsis, i])
                for i in range(num_categorical_variables)
            ]

        else:
            embedded_inputs = []
            regular_inputs = all_inputs[:, :, :num_regular_variables]
            categorical_inputs = None

        # Static inputs
        if len(self._static_input_loc)>0:  # [4]
            static_inputs = [tf.keras.layers.Dense(self.hidden_units, name=f'StaticInputs{i}')(
                regular_inputs[:, 0, i:i + 1]) for i in range(num_regular_variables)
                           if i in self._static_input_loc] + [embedded_inputs[i][:, 0, :]
                   for i in range(num_categorical_variables)
                   if i + num_regular_variables in self._static_input_loc]

            # (?, num_cat_variables, hidden_units) <-  [(?, hidden_units)]
            static_inputs = stack(static_inputs, axis=1)

        else:  # there are not static inputs
            static_inputs = None

        def convert_real_to_embedding(_x, _name=None):
            """Applies linear transformation for time-varying inputs."""
            return TimeDistributed(Dense(self.hidden_units, name=_name))(_x)

        # whether we have and want to use target observations as inputs or not?
        if len(self._input_obs_loc) > 0:
            # Targets
            obs_inputs = stack([          # (?, time_steps, hidden_units, 1)
                convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1], _name='InputObsDense')
                for i in self._input_obs_loc],axis=-1)
        else:
            obs_inputs = None

        wired_embeddings = []
        if num_categorical_variables > 0:
            # Observed (a prioir unknown) inputs
            for i in range(num_categorical_variables):
                if i not in self._known_categorical_input_idx \
                and i not in self._input_obs_loc:
                    e = embeddings[i](categorical_inputs[:, :, i])
                    wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx \
              and i not in self._input_obs_loc:
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1], _name=f'RegularInputsDense{i}')
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            if len(wired_embeddings) > 0:
              unknown_inputs = stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
          unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [  # list of num_outputs all of shape (?, time_steps, hidden_units)
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1], _name=f'KnownRegularInputs')
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [         # []
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        # (?, time_steps, hidden_units, num_outputs)
        known_combined_layer = stack(known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs
