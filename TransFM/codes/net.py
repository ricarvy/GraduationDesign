import tensorflow as tf
import numpy as np

import keras.backend as K
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from Tra

### FM层代码实现
class FMLayer(Layer):
    def __init__(self, output_dim,
                 factor_order,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.factor_order = factor_order
        self.activation = activations.get(activation)
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.w = self.add_weight(name='one',
                                 shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v = self.add_weight(name='two',
                                 shape=(input_dim, self.factor_order),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = K.square(inputs)

        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)

        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim

### Translation层代码实现
class TranslationLayer(Layer):
    def __init__(self, output_dim,
                 factor_order,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.factor_order = factor_order
        self.activation = activations.get(activation)
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.w = self.add_weight(name='one',
                                 shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v = self.add_weight(name='two',
                                 shape=(input_dim, self.factor_order),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = K.square(inputs)

        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)

        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim

### Attention层代码实现
class AttentionNet():
    def __init__(self, data, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.features, self.target = data
        self.net = self.build_net()

    def build_net(self, train=False):
        inputs = Input(shape=self.input_shape)
        x = Embedding(20000, 50)(inputs)
        x = Dropout(0.2)(x)
        x = Conv1D(250, 3, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = FMLayer(200, 100)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.output_shape[0], activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        if train:
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=2,
                      validation_data=(x_test, y_test))

            # model.save_weights('model.h5')
        return model


    def train(self, batch_size=100):
        pass

def fm_model(x_train, x_test, y_train, y_test, train=True):
    inp = Input(shape=(100,))
    x = Embedding(20000, 50)(inp)
    x = Dropout(0.2)(x)
    x = Conv1D(250, 3, padding='valid', activation='relu', strides=1)(x)
    x = GlobalMaxPooling1D()(x)
    x = FMLayer(200, 100)(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    if train:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=2,
                  validation_data=(x_test, y_test))

        model.save_weights('model.h5')
    return model

def get_data():
    x_train = np.arange(0,100).astype('float32').reshape((-1, 100))
    x_test = np.arange(100,200).astype('float32').reshape((-1, 100))
    y_train = np.array([1.])
    y_test = np.array([2.])
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()
    # test_model(x_train, x_test, y_train, y_test, train=True)
    model = fm_model(x_train, x_test, y_train, y_test, train=True)
    print(model.summary())

