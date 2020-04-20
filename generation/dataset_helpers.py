import numpy as np
import os
import glob
import pandas as pd
import tensorflow as tf

# Setting allow_growth for gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model running on CPU")

use_gzip_compression = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
modalities = ['MR_T1_mri', 'MR_T2_mri', 'MR_T1c_mri', 'MR_Flair_mri', 'OT_mri']


# Description on the features contained in the .tfrecord dataset
def get_feature_description(modalities):
    feature_description = lambda mod: {
        mod + '_mri': tf.io.FixedLenFeature([], tf.string),
        mod + '_path': tf.io.FixedLenFeature([], tf.string),

        mod + '_mri_min': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_min_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_max': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_max_src': tf.io.FixedLenFeature([], tf.float32),

        mod + '_mri_lperc': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_hperc': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_hperc_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_mri_lperc_src': tf.io.FixedLenFeature([], tf.float32),

        mod + '_x_dimension': tf.io.FixedLenFeature([], tf.int64),
        mod + '_y_dimension': tf.io.FixedLenFeature([], tf.int64),
        mod + '_z_dimension': tf.io.FixedLenFeature([], tf.int64),
        mod + '_z_dimension_src': tf.io.FixedLenFeature([], tf.int64),
        mod + '_x_dimension_src': tf.io.FixedLenFeature([], tf.int64),
        mod + '_y_dimension_src': tf.io.FixedLenFeature([], tf.int64),

        mod + '_x_origin_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_y_origin_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_z_origin_src': tf.io.FixedLenFeature([], tf.float32),

        mod + '_z_spacing_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_x_spacing_src': tf.io.FixedLenFeature([], tf.float32),
        mod + '_y_spacing_src': tf.io.FixedLenFeature([], tf.float32),

        mod + '_patient': tf.io.FixedLenFeature([], tf.string),
        mod + '_sample_number': tf.io.FixedLenFeature([], tf.string),
        mod + '_patient_grade': tf.io.FixedLenFeature([], tf.string),
        mod + '_location': tf.io.FixedLenFeature([], tf.string),
        mod + '_dataset_version': tf.io.FixedLenFeature([], tf.string),
        mod + '_dataset_name': tf.io.FixedLenFeature([], tf.string),
        mod + '_mri_type': tf.io.FixedLenFeature([], tf.string),
        mod + '_dataset_split': tf.io.FixedLenFeature([], tf.string),
        mod + '_patient_mri_seq': tf.io.FixedLenFeature([], tf.string),
    }
    features = {}
    for mod in modalities:
        features.update(feature_description(mod))
    return features


use_gzip_compression = True


def load_dataset(path, mri_type, center_crop=None, random_crop=None, filter=None, batch_size=32, cache=True,
                 prefetch_buffer=1, shuffle_buffer=128, interleave=1, cast_to=tf.float32, clip_labels_to=0.0,
                 take_only=None, shuffle=True, infinite=False, n_threads=os.cpu_count()):
    def parse_sample(sample_proto):
        parsed = tf.io.parse_single_example(sample_proto, get_feature_description(["OT"] + mri_type))
        # Decoding image arrays

        slice_shape = [parsed['OT_x_dimension'.format(mri_type[0])], parsed['OT_y_dimension'], 1]
        # Decoding the ground truth
        parsed['seg'] = tf.cast(tf.reshape(tf.io.decode_raw(parsed['OT_mri'], tf.float32), shape=slice_shape),
                                dtype=cast_to)
        # Decode each channel and stack in a 3d volume
        stacked_mri = list()
        for mod in mri_type:
            stacked_mri.append(
                tf.cast(tf.reshape(tf.io.decode_raw(parsed['{}_mri'.format(mod)], tf.float32), shape=slice_shape),
                        dtype=cast_to))
        parsed['mri'] = tf.concat(stacked_mri, axis=-1)
        # Clipping the labels if requested
        parsed['seg'] = tf.clip_by_value(parsed['seg'], 0.0, clip_labels_to) if clip_labels_to else parsed['seg']

        # Cropping
        if random_crop or center_crop:
            # Stacking the mri and the label to align the crop shape
            mri_seg = tf.concat([parsed['mri'], parsed['seg']], axis=-1)
            if random_crop:
                random_crop[-1] = mri_seg.shape[-1]
                cropped = tf.image.random_crop(mri_seg, size=random_crop)
            else:
                cropped = tf.image.resize_with_crop_or_pad(mri_seg, center_crop[0], center_crop[1])
            # Splitting back
            parsed['mri'] = cropped[:, :, :len(mri_type)]
            parsed['seg'] = cropped[:, :, len(mri_type):]

        batch0 = process_batch(parsed[modalities[0]])
        batch1 = process_batch(parsed[modalities[1]])
        batch2 = process_batch(parsed[modalities[2]])
        batch3 = process_batch(parsed[modalities[3]])
        batch4 = process_batch(parsed[modalities[4]])

        return batch0, batch1, batch2, batch3, batch4, patient

    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP' if use_gzip_compression else "")
    dataset = dataset.filter(filter) if filter is not None else dataset
    dataset = dataset.take(take_only) if take_only is not None else dataset

    # You should generally cache after loading and preprocessing the data,
    # but before shuffling, repeating, batching and prefetching”
    dataset = dataset.cache() if cache else dataset
    if shuffle and infinite:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer))
    else:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True) if shuffle else dataset
        dataset = dataset.repeat() if infinite else dataset
    dataset = dataset.map(parse_sample, num_parallel_calls=None)
    dataset = dataset.batch(batch_size) if batch_size > 0 else dataset

    if interleave > 1:
        dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(interleave),
                                     cycle_length=n_threads, block_length=interleave, num_parallel_calls=n_threads)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def process_batch(batch):
    batch = tf.io.decode_raw(batch, tf.float32)
    batch = tf.reshape(batch, (1, 180, 180))
    batch = tf.squeeze(batch)
    # The next line of code is to add a padding around the image
    # It allows to have images compatible with the input of the models (256x256).
    batch = tf.pad(batch, tf.constant([[38, 38], [38, 38]]), "CONSTANT")
    batch = tf.expand_dims(batch, axis=2)
    return batch
    # Shape of batch now is (bs, 256, 256, 1): now the batch is ready to be fed to the GAN
