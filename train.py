import tensorflow as tf
import util
import os
import numpy as np
import matplotlib.pyplot as plt

# Task specification.
hr_flist = './data/hr_flist'
lr_flist = './data/lr_flist'

scale = 2

graph_based = False

# Model and data preprocessing.
data_name  = 'data'
model_name = 'model_recurrent_s2_u128_avg_t7-keras'
output_dir = 'ouput'
log_dir    = './training_logs'

# Training hyper parameters
learning_rate = 0.001
batch_size = 16
num_epochs = 3
upsampling_method = 'bicubic'

data = __import__(data_name)
mdl = __import__(model_name)

target, source = data.dataset(hr_flist, lr_flist, scale, upsampling_method, residual=False)

if graph_based:
    tf.compat.v1.disable_eager_execution()

split = int(len(source) * .80)

train = tf.data.Dataset.from_tensor_slices((source[:split],target[:split])).shuffle(3, reshuffle_each_iteration=True)
test  = tf.data.Dataset.from_tensor_slices((source[split:],target[split:])).shuffle(3, reshuffle_each_iteration=True)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

# scheduler is only valuable with a higher number of epochs
if num_epochs > 20:
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5,
                    verbose=1,
                    mode='auto',
                    min_delta=0.0001,
                    cooldown=5,
                    min_lr=0.0001,
                )
    callbacks.append(scheduler)


model = mdl.build_model(scale, training=True, reuse=False)
model.compile(optimizer='adam', loss='mse')

model_history = model.fit(
    train,
    epochs=num_epochs,
    validation_data=test,
    callbacks=callbacks
)

print(model_history.history)

results = model.evaluate(test)

path='./weights/Weights.ckpt'

# save
print("Saving model weights...")
model.save_weights(path)
print("Model weights saved")
