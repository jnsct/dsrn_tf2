import tensorflow as tf
import util
import numpy as np

def dataset(hr_flist, lr_flist, scale, upsampling_method, resize=False, residual=True):
    
    hr_filename_list = np.loadtxt(hr_flist, dtype=str)
    
    lr_filename_list = np.loadtxt(lr_flist, dtype=str)
    
    hr_data = []
    lr_data = []

    
    for hr_filename, lr_filename in zip(hr_filename_list, lr_filename_list):
        
        hr_image = tf.io.decode_image(tf.io.read_file(hr_filename), channels=3)
        hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
        
        hr_image = _rescale(hr_image, 1)
        lr_image = _rescale(hr_image, 1 / scale)

        if residual:
            hr_image = _make_residual(hr_image, lr_image, upsampling_method)

        hr_patches, lr_patches = _make_patches(hr_image, lr_image, scale, resize, upsampling_method)
        hr_patches_rotated, lr_patches_rotated = _make_patches(
            tf.image.rot90(hr_image), tf.image.rot90(lr_image), scale, resize, upsampling_method)
        
        hr_data.extend([hr_patches, hr_patches_rotated])
        lr_data.extend([lr_patches, lr_patches_rotated])
    
    return hr_data, lr_data
    
    
def _rescale(image, target_scale):
    image.set_shape([None, None, 3])
    new_shape = tf.cast(tf.cast(tf.shape(image)[:2],tf.float32) * tf.cast(target_scale,tf.float32),tf.int32)
    return tf.image.resize(image, new_shape, preserve_aspect_ratio=True)
    
def _make_residual(hr_image, lr_image, upsampling_method):
    hr_image = tf.expand_dims(hr_image, 0)
    lr_image = tf.expand_dims(lr_image, 0)
    hr_image_shape = tf.shape(hr_image)[1:3]
    res_image = hr_image - util.get_resize_func(upsampling_method)(lr_image, hr_image_shape)
    return tf.reshape(res_image, [hr_image_shape[0], hr_image_shape[1], 3])

def _make_patches(hr_image, lr_image, scale, resize, upsampling_method):
    hr_image = tf.stack(_flip([hr_image]))
    lr_image = tf.stack(_flip([lr_image]))
    hr_patches = util.image_to_patches(hr_image)
    
    if resize:
        lr_image = util.get_resize_func(upsampling_method)(lr_image, tf.shape(hr_image)[1:3])
        lr_patches = util.image_to_patches(lr_image)
    else:
        lr_patches = util.image_to_patches(lr_image, scale)
    return hr_patches, lr_patches

def _flip(img_list):
    flipped_list = []
    for img in img_list:
        flipped_list.append(
            tf.image.random_flip_up_down(
                tf.image.random_flip_left_right(img), seed=0))
    return flipped_list

# Assuming `util` is a module or class with the necessary functions like `get_resize_func` and `image_to_patches`.
# Make sure to import or define it appropriately.
