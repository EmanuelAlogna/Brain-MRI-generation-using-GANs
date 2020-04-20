import numpy as np
import os
import glob
import pandas as pd
import tensorflow as tf

BATCH_SIZE = 32

def compute_psnr(image1, image2):
    # Compute PSNR over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(image1, tf.float32)  # inp is a numpy.ndarray and im1 is an EagerTensor
    im2 = tf.image.convert_image_dtype(image2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0)
    mean = tf.reduce_mean(tf.boolean_mask((psnr), tf.math.is_finite(psnr)))
    std = tf.math.reduce_std(tf.boolean_mask((psnr), tf.math.is_finite(psnr)))
    # In the computation of mean and std, I'm ignoring the 'nan' and 'inf' values
    # Why 'nan' values? 'nan' happens when there is an image with max_value and min_value = 0.0 so a black image
    # the PSNR would be inf (image is totally similar to the ground truth)
    # be rescaling the image, the max_value and min_value would become nan and so the PSNR

    # It ignores also the 'inf' values, in the case I don't want to normalize

    return mean, std, psnr

def compute_ssim(image1, image2):
    im1 = tf.image.convert_image_dtype(image1, tf.float32)   # inp is a numpy.ndarray and im1 is an EagerTensor
    im2 = tf.image.convert_image_dtype(image2, tf.float32)
    ssim = tf.image.ssim(im1, im2, max_val=1)
    mean = tf.reduce_mean(tf.boolean_mask((ssim), tf.math.is_finite(ssim)))
    std = tf.math.reduce_std(tf.boolean_mask((ssim), tf.math.is_finite(ssim)))
    return mean, std, ssim

# I want to compute, first thing, the MSE between ground truth and generated one. The tf.function gives me a Tensor 32x256x256:
# MSE is computed PIXEL per PIXEL, so per each of the 32 matrix 256x256, I average (1) the values of the 256x256 pixels obtaining
# an array of 32 elements, containing the MSEs of each image belonging to the batch. Then I can average (2) these 32 to have
# I should not average the whole 32x256x256 in one step. The result would have same mean but slightly different std.
# I want first to obtain the MSE of each image... then I average across the batch only to have smth more accurate.

def compute_mse(image1, image2):       # mean squared error
    im1 = tf.image.convert_image_dtype(image1, tf.float32)   # inp is a numpy.ndarray and im1 is an EagerTensor
    im2 = tf.image.convert_image_dtype(image2, tf.float32)
    mse = tf.metrics.mean_squared_error(im1,im2)
    # In this way is possible to do Variable item-assignment with tensors
    mse_per_image = tf.TensorArray(tf.float32, size=BATCH_SIZE)
    for i in range(BATCH_SIZE):
        x = tf.reduce_mean(tf.boolean_mask((mse[i]), tf.math.is_finite(mse[i])))
        mse_per_image = mse_per_image.write(i, x)
    mse_per_image = mse_per_image.stack()
    mean = tf.reduce_mean(tf.boolean_mask((mse_per_image), tf.math.is_finite(mse_per_image)))
    std = tf.math.reduce_std(tf.boolean_mask((mse_per_image), tf.math.is_finite(mse_per_image)))
    return mean, std, mse_per_image

def compute_mse_tumor(image1, image2):  # mean squared error
    im1 = tf.image.convert_image_dtype(image1, tf.float32)  # inp is a numpy.ndarray and im1 is an EagerTensor
    im2 = tf.image.convert_image_dtype(image2, tf.float32)
    squared_difference = tf.math.squared_difference(im1, im2)
    mse_per_image = tf.TensorArray(tf.float32, size=BATCH_SIZE)
    for i in range(BATCH_SIZE):
        non_zero_elements = tf.math.count_nonzero(squared_difference[i], dtype=tf.dtypes.float32)
        sum_over_squared_difference = tf.math.reduce_sum(squared_difference[i])
        x = tf.math.divide(sum_over_squared_difference, non_zero_elements)

        mse_per_image = mse_per_image.write(i, x)
    mse_per_image = mse_per_image.stack()
    mean = tf.reduce_mean(tf.boolean_mask((mse_per_image), tf.math.is_finite(mse_per_image)))
    std = tf.math.reduce_std(tf.boolean_mask((mse_per_image), tf.math.is_finite(mse_per_image)))
    return mean, std, mse_per_image


def compute_psnr_tumor(image1, image2):  # mean squared error
    max_val = 1.0
    im1 = tf.image.convert_image_dtype(image1, tf.float32)  # inp is a numpy.ndarray and im1 is an EagerTensor
    im2 = tf.image.convert_image_dtype(image2, tf.float32)
    squared_difference = tf.math.squared_difference(im1, im2)
    psnr_per_image = tf.TensorArray(tf.float32, size=BATCH_SIZE)
    for i in range(BATCH_SIZE):
        non_zero_elements = tf.math.count_nonzero(squared_difference[i], dtype=tf.dtypes.float32)
        sum_over_squared_difference = tf.math.reduce_sum(squared_difference[i])
        x = tf.math.divide(sum_over_squared_difference, non_zero_elements)

        psnr_val = math_ops.subtract(20 * math_ops.log(max_val) / math_ops.log(10.0),
                                     np.float32(10 / np.log(10)) * math_ops.log(x), name='psnr')
        psnr_per_image = psnr_per_image.write(i, psnr_val)

    psnr_per_image = psnr_per_image.stack()
    mean = tf.reduce_mean(tf.boolean_mask((psnr_per_image), tf.math.is_finite(psnr_per_image)))
    std = tf.math.reduce_std(tf.boolean_mask((psnr_per_image), tf.math.is_finite(psnr_per_image)))
    return mean, std, psnr_per_image