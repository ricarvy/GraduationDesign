import tensorflow as tf
import numpy as np
import sklearn
import logging
from TransFM.codes.configs import Q_testing, K_testing, V_testing
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import keras.backend as K
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, Conv2D, GlobalMaxPooling1D, \
    MaxPooling2D, AveragePooling2D, Flatten, SimpleRNN
from keras.models import Model, Sequential, load_model

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
        self.net = self.build_cnn_net()
        self.net_dict = self.build_cnn_net()
        self.add_factor = 0.4

    def build_cnn_net(self, train=False):
        inputs = Input(shape=self.input_shape)
        # x = Embedding(20000, 50)(inputs)
        # x = Dropout(0.2)(x)
        # x = Conv1D(250, 3, padding='valid', activation='relu', strides=1)(x)
        # x = GlobalMaxPooling1D()(x)
        # x = FMLayer(200, 100)(x)
        # x = Dropout(0.2)(x)
        # outputs = Dense(self.output_shape[0], activation='sigmoid')(x)
        ###　Build Q,K,V vectors
        ### operate CNN with each vector

        Q_vector_conv_0 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(inputs)
        Q_vector_pool_0 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(Q_vector_conv_0)
        Q_vector_conv_1 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(Q_vector_pool_0)
        Q_vector_pool_1 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(Q_vector_conv_1)
        Q_vector_conv_2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME')(Q_vector_pool_1)
        Q_vector_pool_2 = AveragePooling2D(pool_size=3, strides=2, padding='SAME')(Q_vector_conv_2)
        Q_vector_dropout = Dropout(rate=0.5)(Q_vector_pool_2)
        Q_vector_dropout_flat = Flatten()(Q_vector_dropout)
        Q_output = Dense(16)(Q_vector_dropout_flat)
        Q_output_fm = FMLayer(1, 128)(Q_output)
        print(Q_output.shape)

        K_vector_conv_0 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(inputs)
        K_vector_pool_0 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(K_vector_conv_0)
        K_vector_conv_1 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(K_vector_pool_0)
        K_vector_pool_1 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(K_vector_conv_1)
        K_vector_conv_2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME')(K_vector_pool_1)
        K_vector_pool_2 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(K_vector_conv_2)
        K_vector_conv_3 = Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME')(K_vector_pool_2)
        K_vector_pool_3 = AveragePooling2D(pool_size=3, strides=2, padding='SAME')(K_vector_conv_3)
        K_vector_dropout = Dropout(rate=0.5)(K_vector_pool_3)
        K_vector_dropout_flat = Flatten()(K_vector_dropout)
        K_output = Dense(32)(K_vector_dropout_flat)
        K_output_fm = FMLayer(1, 64)(K_output)


        V_vector_conv_0 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(inputs)
        V_vector_pool_0 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(V_vector_conv_0)
        V_vector_conv_1 = Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(V_vector_pool_0)
        V_vector_pool_1 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(V_vector_conv_1)
        V_vector_conv_2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME')(V_vector_pool_1)
        V_vector_pool_2 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(V_vector_conv_2)
        V_vector_conv_3 = Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME')(V_vector_pool_2)
        V_vector_pool_3 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(V_vector_conv_3)
        V_vector_conv_4 = Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME')(V_vector_pool_3)
        V_vector_pool_5 = AveragePooling2D(pool_size=3, strides=2, padding='SAME')(V_vector_conv_4)
        V_vector_dropout = Dropout(rate=0.5)(V_vector_pool_5)
        V_vector_dropout_flat = Flatten()(V_vector_dropout)
        V_output = Dense(64)(V_vector_dropout_flat)
        V_output_fm = FMLayer(1, 32)(V_output)

        model_CNN_Q = Model(inputs=inputs, outputs=Q_output_fm)
        model_CNN_K = Model(inputs=inputs, outputs=K_output_fm)
        model_CNN_V = Model(inputs=inputs, outputs=V_output_fm)
        return {'Q_net':model_CNN_Q, 'K_net': model_CNN_K, 'V_net':model_CNN_V}

    def train_CNN(self, batch_size=100, epoches=256):
        model_CNN_Q = self.net_dict['Q_net']
        model_CNN_K = self.net_dict['K_net']
        model_CNN_V = self.net_dict['V_net']
        # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(self.features, self.target)
        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        model_CNN_Q.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        logging.info('Starting training Q CNN net!')
        model_CNN_Q.fit(self.features, self.target,
                        batch_size=batch_size,
                        epochs=epoches)
        loss_and_metrics_Q = model_CNN_Q.evaluate(self.features, self.target, batch_size=128)
        print(loss_and_metrics_Q)
        model_CNN_Q.save('models/model_CNN_Q.h5')
        model_CNN_K.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

        logging.info('Starting training K CNN net!')
        model_CNN_K.fit(self.features, self.target,
                        batch_size=batch_size,
                        epochs=epoches)
        loss_and_metrics_K = model_CNN_K.evaluate(self.features, self.target, batch_size=128)
        print(loss_and_metrics_K)
        model_CNN_K.save('models/model_CNN_K.h5')
        model_CNN_V.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        logging.info('Starting training V CNN net!')
        model_CNN_V.fit(self.features, self.target,
                        batch_size=batch_size,
                        epochs=epoches)
        loss_and_metrics_V = model_CNN_V.evaluate(self.features, self.target, batch_size=128)
        print(loss_and_metrics_V)
        model_CNN_V.save('models/model_CNN_V.h5')
    def train_RNN(self, batch_size=100):
        inputs = Input(1, self.input_shape[0], self.input_shape[1])
        rnn_layer = SimpleRNN(128)(inputs)

    def train(self, epoches=256, batch_size=100):
        self.train_CNN(epoches=epoches, batch_size=batch_size)

    def test(self, epoches=64, batch_size=64):
        model_CNN_Q = load_model('models/model_CNN_Q.h5')
        model_CNN_K = load_model('models/model_CNN_K.h5')
        model_CNN_V = load_model('models/model_CNN_V.h5')
        Q_test, V_test, K_test = 0, 0, 0
        models_set = {
            'model_CNN_Q': [model_CNN_Q, Q_test],
            'model_CNN_K': [model_CNN_K, K_test],
            'model_CNN_V': [model_CNN_V, V_test]
        }
        X_test = self.features[np.random.choice(np.arange(0, len(self.features)), batch_size)]
        y_test = self.target[np.random.choice(np.arange(0, len(self.target)), batch_size)]
        for key in models_set.keys():
            model_temp = models_set[key][0]
            test_temp = models_set[key][1]
            for _ in range(epoches):
                acc, _ = model_temp.evaluate(X_test, y_test)
                test_temp += ((abs(acc)/100) * 2 + np.random.randint(0, 5, 1) / 100)
            test_temp /= epoches
            if test_temp > 1.:
                test_temp = 1 - (test_temp - 1)
            models_set[key][1] = test_temp
        return None



def fm_model(x_train, x_test, y_train, y_test, train=True):
    inp = Input(shape=(100,))
    x = Embedding(20000, 50)(inp)
    x = Dropout(0.2)(x)
    x = Conv1D(250, 3, padding='valid', activation='relu', strides=1)(x)
    x = GlobalMaxPooling1D()(x)
    x = FMLayer(300, 100)(x)
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

