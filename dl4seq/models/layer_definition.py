#! -*- coding: utf-8 -*-


from .backend import keras


class MyDot(keras.layers.Layer):  # The parameters are (inputs, output_dim) only change the last dimension of the input
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyDot, self).__init__(**kwargs)

    def build(self, input_shape):

        # it seems in tf-keras we have fully define the shape and we can not define shape as (-1, 20)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',  # 'uniform'
                                      name='kernel',
                                      trainable=True)
        super(MyDot, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return keras.backend.dot(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class MyTranspose(keras.layers.Layer):  # The parameters are (inputs, axis) to change the dimension
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(MyTranspose, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyTranspose, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return keras.backend.permute_dimensions(inputs, pattern=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape[self.axis[0]], input_shape[self.axis[1]], input_shape[self.axis[2]]
