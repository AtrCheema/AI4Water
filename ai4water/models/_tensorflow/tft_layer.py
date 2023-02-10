__all__ = ['TemporalFusionTransformer']

"""
This file contain most of the code from
https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
The TemporalFusionTransformer class has been modified so that it can be used as regular layer and without static,
categorical, observation and future inputs. It has also been modified to use 1D CNN as
an alternative of LSTM
The original code had Apache Licence, Version 2.0 although a lot of
the code has been modified here. http://www.apache.org/licenses/LICENSE-2.0
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import TimeDistributed, Dense, InputLayer, Embedding, Conv1D, Flatten

from .utils import concatenate, gated_residual_network, add_and_norm, apply_gating_layer, get_decoder_mask
from .utils import InterpretableMultiHeadAttention


# Layer definitions.
stack = tf.keras.backend.stack


class TemporalFusionTransformer(tf.keras.layers.Layer):
    """
    Implements the model of https://arxiv.org/pdf/1912.09363.pdf
    This layer applies variable selection three times. First on static inputs,
    then on encoder inputs and then on decoder inputs. The corresponding weights
    are called `static_weights`, `historical_weights` and `future_weights` respectively.

    1, 11, 21, 31, a
    2, 12, 22, 32, b
    3, 13, 23, 33, c
    4, 14, 24, 34, d

    Parameters
    ---------
        hidden_units : int
            determines the depth/weight matrices size in TemporalFusionTransformer.
        num_encoder_steps : int
            lookback steps used in the model.
        num_heads : int
            must be>=1, number of attention heads to be used in MultiheadAttention layer.
        num_inputs : int
            number of input features
        total_time_steps : int
            greater than num_encoder_steps, This is sum of lookback steps + forecast length.
            Forecast length is the number of horizons to be predicted.
        known_categorical_inputs : list
            a,b,c
        input_obs_loc :
        unknown_inputs :
        static_inputs :
        static_input_loc None/list:
            location of static inputs
        category_counts : list
            Number of categories per categorical variable
        use_cnn : bool
            whether to use cnn or not. If not, then lstm will be used otherwise
            1D CNN will be used with "causal" padding.
        kernel_size : int
            kernel size for 1D CNN. Only valid if use_cnn is True.
        use_cudnn : bool
            default False, Whether to use Keras CuDNNLSTM or standard LSTM layers
        dropout_rate : float
            default 0.1, >=0 and <=1 amount of dropout to be used at GRNs.
        future_inputs : bool
            whether the given data contains futre known observations or not.
        return_attention_components bool:
            If True, then this layer (upon its call) will return outputs + attention
            componnets. Attention components are dictionary consisting of following keys
            and their values as numpy arrays.

        return_sequences bool:
            if True, then output and attention weights will consist of  encoder_lengths/lookback
            and decoder_length/forecast_len. Otherwise predictions for only decoder_length will be
            returned.

    Example
    -------
        >>> params = {'num_inputs': 3, 'total_time_steps': 192, 'known_regular_inputs': [0, 1, 2]}
        >>> output_size = 1
        >>> quantiles = [0.25, 0.5, 0.75]
        >>> layers = {
        >>>   "Input": {"config": {"shape": (params['total_time_steps'], params['num_inputs']), 'name': "Model_Input"}},
        >>>   "TemporalFusionTransformer": {"config": params},
        >>>   "lambda": {"config": tf.keras.layers.Lambda(lambda _x: _x[Ellipsis, -1, :])},
        >>>   "Dense": {"config": {"units": output_size * len(quantiles)}},
        >>>   'Reshape': {'target_shape': (3, 1)}}

    """
    def __init__(
            self,
            hidden_units: int,
            num_encoder_steps: int,
            num_heads: int,
            num_inputs: int,
            total_time_steps: int,
            known_categorical_inputs,
            static_input_loc,
            category_counts,
            known_regular_inputs,
            input_obs_loc,
            use_cnn: bool = False,
            kernel_size: int = None,
            # stack_size:int = 1,
            use_cudnn: bool = False,
            dropout_rate: float = 0.1,
            future_inputs: bool = False,
            return_attention_components: bool = False,
            return_sequences: bool = False,
            **kwargs
    ):

        if use_cudnn:
            assert kernel_size is not None

        self.time_steps = total_time_steps
        self.input_size = num_inputs
        self.use_cnn = use_cnn
        self.kernel_size = kernel_size
        self._known_regular_input_idx = known_regular_inputs  # [1,2,3]
        self._input_obs_loc = input_obs_loc  # [0]
        self._static_input_loc = static_input_loc  # [3,4]
        self.category_counts = category_counts  # [2, 2]
        self._known_categorical_input_idx = known_categorical_inputs

        # Network params
        self.use_cudnn = use_cudnn  # Whether to use GPU optimised LSTM
        self.hidden_units = int(hidden_units)
        self.dropout_rate = float(dropout_rate)
        self.encoder_steps = num_encoder_steps  # historical steps/lookback steps
        self.num_heads = int(num_heads)
        # self.num_stacks= int(stack_size) # todo

        self.future_inputs = future_inputs
        self.return_attention_components = return_attention_components
        self.return_sequences = return_sequences

        super().__init__(**kwargs)

    def __call__(self, alle_inputs, *args, **kwargs):
        """Returns graph defining layers of the TFT.
        """

        # Size definitions.
        encoder_steps = self.encoder_steps  # encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(alle_inputs)

        # known_combined_layer.shape = (num_examples, time_steps, hidden_units, num_outputs)
        # obs_inputs.shape = (num_examples, time_steps, hidden_units, 1)
        # static_inputs.shape = (num_examples, num_cat_variables, hidden_units)

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
                # (num_examples, encoder_steps, hidden_units, 4) <- (num_examples, encoder_steps, hidden_units, num_outputs) | (num_examples, encoder_steps, hidden_units, 1)
                historical_inputs = concatenate([
                    known_combined_layer[:, :encoder_steps, :],
                    obs_inputs[:, :encoder_steps, :]], axis=-1, name="historical_inputs")
            else:
                historical_inputs = known_combined_layer[:, :encoder_steps, :]

        if self.future_inputs:
            assert self.time_steps - self.encoder_steps > 0
            # Isolate only known future inputs.
            future_inputs = known_combined_layer[:, encoder_steps:, :]  # (num_examples, 24, hidden_units, num_outputs)
        else:
            assert self.time_steps == self.encoder_steps
            future_inputs = None

        def static_combine_and_mask(embedding):
            """Applies variable selection network to static inputs.

            Args:
              embedding: Transformed static inputs (num_examples, num_cat_variables, hidden_units)

            Returns:
              Tensor output for variable selection network
            """

            # Add temporal features
            _, num_static, _ = embedding.get_shape().as_list()

            flatten = tf.keras.layers.Flatten()(embedding)  # (num_examples, hidden_units*num_cat_variables)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = gated_residual_network(  # (num_examples, num_cat_variables)
                flatten,
                self.hidden_units,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None,
                name='GRN_static'
            )

            # (num_examples, num_cat_variables)
            sparse_weights = tf.keras.layers.Activation('softmax', name='sparse_static_weights')(mlp_outputs)
            sparse_weights = K.expand_dims(sparse_weights, axis=-1)  # (num_examples, num_cat_variables, 1)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(  # e.shape = (num_examples, 1, hidden_units)
                    embedding[:, i:i + 1, :],
                    self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                    name=f'GRN_static_{i}'
                )
                trans_emb_list.append(e)

            # (num_examples, num_cat_variables, hidden_units)
            transformed_embedding = concatenate(trans_emb_list, axis=1, name="transfomred_embedds")

            # (num_examples, num_cat_variables, hidden_units)
            combined = tf.keras.layers.Multiply(name="StaticWStaticEmb")(
                [sparse_weights, transformed_embedding])

            static_vec = K.sum(combined, axis=1)  # (num_examples, hidden_units)

            return static_vec, sparse_weights

        static_context_state_h = None
        static_context_state_c = None
        static_context_variable_selection = None
        static_context_enrichment = None
        static_weights = None

        if static_inputs is not None:
            static_encoder, static_weights = static_combine_and_mask(static_inputs)

            # static_encoder.shape = (num_examples, hidden_units)
            # static_weights.shape = (num_examples, num_cat_variables, 1)
            static_context_variable_selection = gated_residual_network(  # (num_examples, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GNR_st_cntxt_var_select"
            )
            static_context_enrichment = gated_residual_network(  # (num_examples, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_enrich"
            )
            static_context_state_h = gated_residual_network(  # (num_examples, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_h"
            )
            static_context_state_c = gated_residual_network(  # (num_examples, hidden_units)
                static_encoder,
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                name="GRN_st_cntxt_c"
            )

        def lstm_combine_and_mask(embedding, static_context, _name=None):
            """Apply temporal variable selection networks.

            Args:
              embedding: Transformed inputs. (num_examples, time_steps, hidden_units, num_inputs)
              # time_steps can be either encoder_steps or decoder_steps.
              # num_inputs can be either encoder_inputs or decoder_inputs
              static_context:
              _name: name of encompassing layers

            Returns:
              Processed tensor outputs. (num_examples, time_steps, hidden_units)
            """

            # Add temporal features
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

            flatten = K.reshape(embedding,  # (num_examples, time_steps, num_inputs*hidden_units)
                                [-1, time_steps, embedding_dim * num_inputs])

            if static_context is not None:
                _expanded_static_context = K.expand_dims(  # (num_examples, 1, hidden_units)
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

            # mlp_outputs.shape (num_examples, time_steps, num_inputs)
            # static_gate.shape (num_examples, time_steps, num_inputs)
            # sparse_weights (num_examples, time_steps, num_inputs)
            sparse_weights = tf.keras.layers.Activation('softmax',    # --> (num_examples, time_steps, num_inputs)
                                                        name=f'sparse_{_name}_weights_softmax')(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2)  # (num_examples, time_steps, 1, num_inputs)

            # Non-linear Processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(  # --> (num_examples, time_steps, hidden_units)
                    embedding[Ellipsis, i],   # feeding (num_examples, time_steps, hidden_units) as input
                    self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                    name=f'GRN_with_{_name}_for_{i}'
                )
                trans_emb_list.append(grn_output)

            # (num_examples, time_steps, hidden_units, num_inputs)
            transformed_embedding = stack(trans_emb_list, axis=-1)

            # --> (num_examples, time_steps, hidden_units, num_inputs)
            combined = tf.keras.layers.Multiply(name=f'sparse_and_transform_{_name}')(
                [sparse_weights, transformed_embedding])
            temporal_ctx = K.sum(combined, axis=-1)  # (num_examples, time_steps, hidden_units)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_weights, historical_gate = lstm_combine_and_mask(
            historical_inputs,
            static_context_variable_selection,
            _name='history'
        )
        # historical_features.shape = (num_examples, encoder_steps, hidden_units)
        # historical_flags (num_examples, encoder_steps, 1, 4)
        future_features = None
        future_weights = None
        if future_inputs is not None:
            future_features, future_weights, future_gate = lstm_combine_and_mask(
                future_inputs,
                static_context_variable_selection,
                _name='future')

        # future_features = (num_examples, decoder_length, hidden_units)
        # future_flags = (num_examples, decoder_length, 1, num_outputs)

        initial_states = None
        if static_context_state_h is not None:
            initial_states = [static_context_state_h, static_context_state_c]

        if self.use_cnn:
            history_lstm = build_cnn(historical_features,
                                     self.hidden_units,
                                     self.kernel_size,
                                     return_state=True,
                                     _name="history",
                                     use_cudnn=self.use_cudnn)
        else:
            lstm = get_lstm(self.hidden_units, return_state=True, _name="history", use_cudnn=self.use_cudnn)
            history_lstm, state_h, state_c = lstm(historical_features, initial_state=initial_states)
        # history_lstm = (num_examples, encoder_steps, hidden_units)

        if future_features is not None:
            if self.use_cnn:
                future_lstm = build_cnn(future_features, self.hidden_units, self.kernel_size,
                                        return_state=False, _name="Future", use_cudnn=self.use_cudnn)

            else:
                lstm = get_lstm(self.hidden_units, return_state=False, _name="Future", use_cudnn=self.use_cudnn)
                future_lstm = lstm(future_features, initial_state=[state_h, state_c])

            # future_lstm = (num_examples, decoder_length, hidden_units)

            lstm_output = concatenate([history_lstm, future_lstm],
                                      axis=1,
                                      name='history_plus_future_lstm')  # (num_examples, time_steps, hidden_units)
            # Apply gated skip connection
            input_embeddings = concatenate([historical_features, future_features],
                                           axis=1,
                                           name="history_plus_future_embeddings"
                                           )  # (num_examples, time_steps, hidden_units)
        else:
            lstm_output = history_lstm
            input_embeddings = historical_features

        lstm_output, _ = apply_gating_layer(  # (num_examples, time_steps, hidden_units)
            lstm_output, self.hidden_units, self.dropout_rate, activation=None, name='GatingOnLSTM')
        # (num_examples, time_steps, hidden_units)
        temporal_feature_layer = add_and_norm([lstm_output, input_embeddings], name='AfterLSTM')

        # Static enrichment layers
        expanded_static_context = None
        if static_context_enrichment is not None:
            # (num_examples, 1, hidden_units)
            expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)

        atten_input, _ = gated_residual_network(  # (num_examples, time_steps, hidden_units)
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

        mask = get_decoder_mask(atten_input)
        # in some implementation cases, queries contain only decoder_length part but since official google repo
        # used all the inputs i.e. encoder+decoder part, we are doing so as well.
        # This is more useful in cases since transformer layer can be used as many to many.
        # Thus current behaviour is similar to `return_sequences=True` of LSTM.
        if self.return_sequences:
            queries = atten_input
        else:
            queries = atten_input[:, self.encoder_steps:]

        # queries (batch_size, time_steps, hidden_units
        # atten_input (batch_size, time_steps, hidden_units
        atten_output, self_att = self_attn_layer(queries,
                                                 atten_input, atten_input, mask=mask)

        # atten_output (batch_size, time_steps, hidden_units)
        # self_att (num_heads, batch_size, time_steps, time_steps
        # x =  (num_examples, time_steps, hidden_units)
        atten_output, _ = apply_gating_layer(  # # x =  (num_examples, time_steps, hidden_units)
            atten_output,
            self.hidden_units,
            dropout_rate=self.dropout_rate,
            activation=None,
            name="GatingOnX"
        )
        # # x =  (num_examples, time_steps, hidden_units)
        atten_output = add_and_norm([atten_output, queries], name="XAndEnriched")

        # Nonlinear processing on outputs
        decoder = gated_residual_network(  # # x =  (num_examples, time_steps, hidden_units)
            atten_output,
            self.hidden_units,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="NonLinearityOnOut"
        )

        # Final skip connection
        decoder, _ = apply_gating_layer(decoder, self.hidden_units, activation=None,
                                        name="FinalSkip")  # # x =  (num_examples, time_steps, hidden_units)

        # (num_examples, time_steps, hidden_units)
        transformer_output = add_and_norm([decoder, temporal_feature_layer], name="DecoderAndTempFeature")

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,  # (num_atten_heads, num_examples, time_steps, time_steps)
            # Static variable selection weights  # (num_examples, 1)
            'static_variable_selection_weights': static_weights[Ellipsis, 0] if static_weights is not None else None,
            # Variable selection weights of past inputs  # (num_examples, encoder_steps, input_features)
            'encoder_variable_selection_weights': historical_weights[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            # (num_examples, decoder_steps, input_features)
            'decoder_variable_selection_weights': future_weights[Ellipsis, 0, :] if future_weights is not None else None
        }

        self.attention_components = attention_components

        if self.return_attention_components:
            return transformer_output, attention_components

        return transformer_output

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

            # regular_inputs
            regular_inputs, categorical_inputs = all_inputs[:, :, :num_regular_variables], \
                                                 all_inputs[:, :, num_regular_variables:]

            # regular_inputs (num_examples, time_steps, 4)
            # categorical_inputs (num_examples, time_steps, num_static_inputs)
            # list of lengnth=(num_static_inputs) with shape (num_examples, time_steps, hidden_units)
            embedded_inputs = [
                embeddings[i](categorical_inputs[Ellipsis, i])
                for i in range(num_categorical_variables)
            ]

        else:
            embedded_inputs = []
            # --> (num_examples, total_time_steps, num_inputs)
            regular_inputs = all_inputs[:, :, :num_regular_variables]
            categorical_inputs = None

        # Static inputs
        if len(self._static_input_loc) > 0:  # [4]
            static_inputs = [tf.keras.layers.Dense(self.hidden_units, name=f'StaticInputs{i}')(
                regular_inputs[:, 0, i:i + 1]) for i in range(num_regular_variables)
                           if i in self._static_input_loc] + [embedded_inputs[i][:, 0, :]
                   for i in range(num_categorical_variables)
                   if i + num_regular_variables in self._static_input_loc]

            # (num_examples, num_cat_variables, hidden_units) <-  [(num_examples, hidden_units)]
            static_inputs = stack(static_inputs, axis=1)

        else:  # there are not static inputs
            static_inputs = None

        def convert_real_to_embedding(_x, _name=None):
            """Applies linear transformation for time-varying inputs."""
            return TimeDistributed(Dense(self.hidden_units, name=_name))(_x)

        # whether we have and want to use target observations as inputs or not?
        if len(self._input_obs_loc) > 0:
            # Targets
            obs_inputs = stack([          # (num_examples, time_steps, hidden_units, 1)
                convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1], _name='InputObsDense')
                for i in self._input_obs_loc], axis=-1)
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
        known_regular_inputs = [  # list of tensors all of shape (num_examples, total_time_steps, hidden_units)
            # feeding (num_examples, total_time_steps, 1) at each loop
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1], _name=f'KnownRegularInputs')
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [         # []
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        # (num_examples, time_steps, hidden_units, num_outputs)
        known_combined_layer = stack(known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs


def build_cnn(inputs, filters, kernel_size, return_state, _name, use_cudnn=True):

    cnn = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", name=_name)
    cnn_output = cnn(inputs)

    return cnn_output


# LSTM layer
def get_lstm(hidden_units, return_state, _name=None, use_cudnn=True):
    """Returns LSTM cell initialized with default parameters."""
    if use_cudnn:
        lstm = tf.keras.layers.CuDNNLSTM(
            hidden_units,
            return_sequences=True,
            return_state=return_state,
            stateful=False,
            name=_name
        )
    else:
        lstm = tf.keras.layers.LSTM(
            hidden_units,
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
