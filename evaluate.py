import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Task specification.
hr_flist = './data/hr_flist'
lr_flist = './data/lr_flist'
scale = 2

# Model and data preprocessing.
data_name  = 'data'
model_name = 'model_recurrent_s2_u128_avg_t7-keras'
output_dir = 'ouput'
upsampling_method = 'bicubic'

data = __import__(data_name)
mdl = __import__(model_name)

model = mdl.build_model(scale, training=False, reuse=False)
model.compile(optimizer='adam', loss='mse')

path='./weights/Weights.ckpt'
model.load_weights(path)

target, source = data.dataset(hr_flist, lr_flist, scale, upsampling_method, residual=False)
target, source = np.array(target), np.array(source)

target, source = data.dataset(hr_flist, lr_flist, scale, upsampling_method, residual=False)
split = int(len(source) * .80)
test  = tf.data.Dataset.from_tensor_slices((source[split:],target[split:]))

print('Evaluating Model...')
results = model.evaluate(test)

print('Making Predictions...')
predictions = model.predict(tf.data.Dataset.from_tensor_slices((source[split:])))

# generate a random sample of indexes to visually check results
rng = np.random.default_rng()
numbers = rng.choice(len(predictions), size=10, replace=False)

t = np.concatenate(target[split:], axis=0)
for i in numbers:

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions[i])
    plt.title('Prediction')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(t[i])
    plt.title('Target')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
