import tensorflow as tf 

from tensorflow.keras.layers import (Input, 
                                     Layer, 
                                     LSTMCell, 
                                     StackedRNNCells, 
                                     RNN,
                                     Dense)
from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow.keras import Model

from core.masking import create_mask

class Encoder(Model):
    def __init__(self, num_units, num_layers, zdim, dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        rnn_cells = [LayerNormLSTMCell(num_units, dropout=dropout) for _ in range(num_layers)]
        stacked_lstm = StackedRNNCells(rnn_cells)
        self.lstm_layer = RNN(stacked_lstm)
        self.dense = Dense(zdim, activation='tanh')

    def call(self, inputs, training=False):
        x, mask = inputs
        x = self.lstm_layer(x, mask=mask, training=training)
        x = self.dense(x)
        return x 

class Decoder(Model):
    def __init__(self, num_units, num_layers, voc_size, dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        rnn_cells = [LayerNormLSTMCell(num_units, dropout=dropout) for _ in range(num_layers)]
        stacked_lstm = StackedRNNCells(rnn_cells)
        self.lstm_layer = RNN(stacked_lstm, return_sequences=True)
        self.dense = Dense(voc_size)

    def call(self, inputs, training=False):
        x = self.lstm_layer(inputs, training=training)
        x = self.dense(x)
        return x 

class RAE(Model):
    def __init__(self, num_units, num_layers, voc_size, zdim, dropout, **kwargs):
        super(RAE, self).__init__(**kwargs)
        self.voc_size = voc_size
        self.encoder = Encoder(num_units, num_layers, zdim, dropout, name='Encoder')
        self.decoder = Decoder(num_units, num_layers, voc_size, dropout, name='Decoder')      

    def model(self, batch_size):
        serie_1  = Input(shape=(100, 1), batch_size=batch_size, name='Serie1')
        serie_2  = Input(shape=(100, 1), batch_size=batch_size, name='Mask')
        data = (serie_1, serie_2)
        return Model(inputs=data, outputs=self.call(data))

    def call(self, inputs, training=False):
        inp, mask =  inputs
        x = self.encoder(inputs, training=training)
        
        batch_size = tf.shape(inp)[0]
        time_steps = tf.shape(inp)[1]

        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, time_steps, 1])
        steps_rec = tf.reshape(tf.range(time_steps, dtype=tf.float32), [1, time_steps, 1])
        steps_rec = tf.tile(steps_rec, [batch_size, 1, 1])  
        x = tf.concat([steps_rec, x], 2)
        x = self.decoder(x, training=training)
        return x 

    def train_step(self, data):
        x, l, _, _, _ = data
        mask = create_mask(x, l)
        with tf.GradientTape() as tape:
            y_pred = self((x, mask), training=True)
            y_true = tf.concat([x, tf.cast(tf.expand_dims(mask, 2), dtype=tf.float32)], 2)
            t_loss = self.compiled_loss(y_true, y_pred)
            
        gradients = tape.gradient(t_loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, l, y_true, _, _ = data
        mask = create_mask(x, l)
        with tf.GradientTape() as tape:
            y_pred = self((x, mask), training=False)
            y_true = tf.concat([x, tf.cast(tf.expand_dims(mask, 2), dtype=tf.float32)], 2)
            t_loss = self.compiled_loss(y_true, y_pred)
            
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x, l, y_true, _, _ = data
        mask = create_mask(x, l)
        y_pred = self((x, mask), training=False)
        y_true = tf.concat([x, tf.cast(tf.expand_dims(mask, 2), dtype=tf.float32)], 2)
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}