import numpy as np
import SimpleITK as sitk
import os
import glob
import pandas as pd
import tensorflow as tf

datasets_path_pattern = {
    'brats2015-Test-all': '../../datasets/BRATS2015/Testing/*/*/*/*.mha',
    'brats2015-Train-all': '../../datasets/BRATS2015/BRATS2015_Training/*/*/*/*.mha',
    'BD2Decide-T1T2': '../../datasets/BD2Decide/T1T2/*/*.mha'
}
output_stat_path = '../datasets/meta/'

use_gzip_compression = True

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Description on the features contained in the .tfrecord dataset
def get_feature_description(modalities):

    feature_description =lambda mod : {
                            mod+'_mri': tf.io.FixedLenFeature([], tf.string),
                            mod+'_path': tf.io.FixedLenFeature([], tf.string),
        
                            mod+'_mri_min': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_mri_min_src': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_mri_max': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_mri_max_src': tf.io.FixedLenFeature([], tf.float32),

                            mod+'_mri_lperc': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_mri_hperc': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_mri_hperc_src': tf.io.FixedLenFeature([], tf.float32),   
                            mod+'_mri_lperc_src': tf.io.FixedLenFeature([], tf.float32),
        
                            mod+'_x_dimension': tf.io.FixedLenFeature([], tf.int64),
                            mod+'_y_dimension': tf.io.FixedLenFeature([], tf.int64),
                            mod+'_z_dimension': tf.io.FixedLenFeature([], tf.int64),        
                            mod+'_z_dimension_src': tf.io.FixedLenFeature([], tf.int64),
                            mod+'_x_dimension_src': tf.io.FixedLenFeature([], tf.int64),
                            mod+'_y_dimension_src': tf.io.FixedLenFeature([], tf.int64),
        
                            mod+'_x_origin_src': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_y_origin_src': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_z_origin_src': tf.io.FixedLenFeature([], tf.float32),

                            mod+'_z_spacing_src': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_x_spacing_src': tf.io.FixedLenFeature([], tf.float32),
                            mod+'_y_spacing_src': tf.io.FixedLenFeature([], tf.float32),

                            mod+'_patient': tf.io.FixedLenFeature([], tf.string),                            
                            mod+'_sample_number': tf.io.FixedLenFeature([], tf.string),
                            mod+'_patient_grade': tf.io.FixedLenFeature([], tf.string),
                            mod+'_location': tf.io.FixedLenFeature([], tf.string),
                            mod+'_dataset_version': tf.io.FixedLenFeature([], tf.string),
                            mod+'_dataset_name': tf.io.FixedLenFeature([], tf.string),
                            mod+'_mri_type': tf.io.FixedLenFeature([], tf.string),        
                            mod+'_dataset_split': tf.io.FixedLenFeature([], tf.string),
                            mod+'_patient_mri_seq': tf.io.FixedLenFeature([], tf.string),
                          }
    features = {}
    for mod in modalities:
        features.update(feature_description(mod))
    return features

def preprocess_mha(file_path, dataset_min, dataset_max, center_crop=[180,180,128], normalize_to='mri', percentiles=[2, 98]):
    '''
    Read a .mha file, preprocess the results and extract a series of data from the filename (according to dataset organization):
    'patient_grade': grade of patient condition
    'dataset_version': version of the dataset the sample first appeared
    'patient_number': code of the patient in the current dataset
    'patient_mri_seq': sequence number of the sample for the current patient. It's 1 if this is the first sample for the patient.
    'location': Area that the sample represents
    'mri_type': Sequencing type of the MRI
    'dataset_split': Wether if the sample belong to training or testing dataset. Unknown if the dataset hasn't been splitted already
    :param file_path: Path of a single .mha file. has to respect dataset_path_pattern
    :return:
    '''
    meta = dict()
    # Parsing meta from paths
    if 'BRATS2015' in file_path:
        splitted = file_path.split('/')
        meta['dataset_name'] = 'BRATS2015'
        meta['patient_grade'] = 'HIGH' if splitted[-4]=='HGG' else 'LOW' if splitted[-4]=='LGG' else 'unknown'
        _, meta['dataset_version'], meta['patient'], meta['patient_mri_seq'] = splitted[-3].split("_")
        _, meta['location'], _, _, meta['mri_type'], meta['sample_number'], ext = splitted[-1].split(".")
        meta['dataset_split'] = 'training' if 'Training' in file_path else 'testing' if 'Testing' in file_path else 'unknown'
        meta['path'] = file_path
    elif 'BD2Decide' in file_path:
        splitted = file_path.split('/')
        meta['dataset_name'] = 'BD2Decide'
        meta['patient_grade'] = 'unknown'
        meta['dataset_version'] = 'BD2Decide'
        meta['patient'] = splitted[-2]
        meta['patient_mri_seq'] = splitted[-2].split("_")[-1]
        meta['location'] = "Head-Neck"
        meta['sample_number'] = 'unknown'
        filename = splitted[-1].split('.')[0]
        meta['mri_type'] = 'T1' if 'T1' in filename else 'T2' if 'T2' in filename else 'OT' if 'ROI' in filename else filename
        meta['dataset_split'] = 'training' if 'Training' in file_path else 'testing' if 'Testing' in file_path else 'unknown'
        meta['path'] = file_path
    else:
        raise NotImplementedError("Unknown dataset. Please implement how to extract information from the file_path")
    
    # Parsing meta from .mha
    image, origin, spacing = load_itk(file_path)
    
    
    # Keeping original values (before cropping/normalization)
    meta['z_dimension_src'] = image.shape[0]
    meta['x_dimension_src'] = image.shape[1]
    meta['y_dimension_src'] = image.shape[2]
    meta['z_origin_src'] = origin[0]
    meta['x_origin_src'] = origin[1]
    meta['y_origin_src'] = origin[2]
    meta['z_spacing_src'] = spacing[0]
    meta['x_spacing_src'] = spacing[1]
    meta['y_spacing_src'] = spacing[2]
    meta['mri_max_src'] = image.max().astype(np.float32)
    meta['mri_min_src'] = image.min().astype(np.float32)
    meta['mri_lperc_src'], meta['mri_hperc_src'] = np.percentile(image, percentiles).astype(np.float32)
    
    # Preprocessing data

    if center_crop:
        image = image[int(image.shape[0] / 2 - center_crop[2] / 2):int(image.shape[0] / 2 + center_crop[2] / 2),
              int(image.shape[1] / 2 - center_crop[0] / 2):int(image.shape[1] / 2 + center_crop[0] / 2),
              int(image.shape[2] / 2 - center_crop[1] / 2):int(image.shape[2] / 2 + center_crop[1] / 2)]
        
    if normalize_to and meta['mri_type'] != "OT": # We don't normalize label maps here
        if normalize_to == 'dataset':
            max_value = dataset_max
            min_value = dataset_min
        if normalize_to == 'mri':
            min_value, max_value = np.percentile(image, percentiles).astype(np.float32)
        if normalize_to == 'slice':
            raise NotImplementedError()

        image = rescale(image, min_value, max_value, 0, 1)
        image = np.clip(image, 0, 1)

    # Computing statistics at mri level
    meta['mri_max'] = image.max().astype(np.float32)
    meta['mri_min'] = image.min().astype(np.float32)
    meta['mri_lperc'], meta['mri_hperc'] = np.percentile(image, percentiles).astype(np.float32)
    meta['z_dimension'] = image.shape[0]
    meta['x_dimension'] = image.shape[1]
    meta['y_dimension'] = image.shape[2]
    meta['mri'] = image
    return meta


def preprocess_slice(mri_meta, dataset_name, normalize_to=None, percentiles=[2,98]):
    slices = []
    if 'brats2015' in dataset_name.lower() or 'bd2decide' in dataset_name.lower():
        for z in range(mri_meta['OT']['mri'].shape[0]):
            meta = dict()
            for modality in mri_meta.keys():
                # For each modality we have to duplicate the meta keys (except the data)
                meta.update({'{}_{}'.format(modality, k):mri_meta[modality][k] for k in mri_meta[modality] if k != 'mri'})
                mri_slice = mri_meta[modality]['mri'][z,...]

                meta['{}_slice_lperc'.format(modality)], meta['{}_slice_hperc'.format(modality)] = np.percentile(mri_slice, percentiles).astype(np.float32)
                meta['{}_slice_min'.format(modality)], meta['{}_slice_max'.format(modality)] = mri_slice.min().astype(np.float32), mri_slice.max().astype(np.float32)

                if normalize_to == 'slice' and modality != "OT": # we don't normalize labels
                    # rescaling
                    mri_slice = rescale(mri_slice, meta['{}_slice_lperc'.format(modality)], meta['{}_slice_hperc'.format(modality)], 0, 1)
                    mri_slice = np.clip(mri_slice, 0, 1)
                meta['{}_{}'.format(modality, 'mri')] = mri_slice.astype(np.float32).tobytes()
            slices.append(meta)
    else:
        raise NotImplementedError
    return slices


def load_itk(filename):
    ''' Read an .mha image and returns a numpy array, the origin coordinates and the voxel sizes in mm '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,x,y
    image = sitk.GetArrayFromImage(itkimage)
    # Here we have (z, y, x).
    image = image.transpose((0, 2, 1))

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(itkimage.GetOrigin())[[2,0,1]]

    # Read the spacing along each dimension
    spacing = np.array(itkimage.GetSpacing())[[2,0,1]]

    return image, origin, spacing




def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if type(value) == str:
      value = value.encode('utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value, flatten=False):
  """Returns an int64_list from a bool / enum / int / uint."""
  if flatten:
      value = value.flatten()
  else:
      value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize(value):
    '''
    Selects the right _xxx_feature function for a given value
    :return:
    '''
    if type(value) in [float, np.float, np.float16, np.float32, np.float64]:
        return _float_feature(value)
    if type(value) in [int, np.int, np.int16, np.int32, np.uint, np.uint8, np.uint16, np.uint32, np.int64]:
        return _int64_feature(value)
    if type(value) in [str, np.str, bytes]:
        return _bytes_feature(value)

def split_train_validation(groups_pattern, ratio_train=0.9):

    '''
    Split in train/validation sets according to SegAN paper and BRATS dataset.
    :param groups_pattern: A list of patterns that define the groups. For example ['BRATS2015_Training/HGG/*/', 'BRATS2015_Training/LGG/*/'] to stratify according to the patient grade in BRATS
    '''
    import glob
    from random import shuffle
    train = list()
    valid = list()
    
    for gp in groups_pattern:
        filenames = glob.glob(gp)
        shuffle(filenames)
        train += filenames[0:int(ratio_train*len(filenames))]
        valid += filenames[int(ratio_train*len(filenames)):]
    return train, valid

def split_train_validation_test(groups_pattern, ratio_train=0.6, ratio_valid=0.5):

    '''
    Split in train/validation sets according to SegAN paper and BRATS dataset.
    :param groups_pattern: A list of patterns that define the groups. For example ['BRATS2015_Training/HGG/*/', 'BRATS2015_Training/LGG/*/'] to stratify according to the patient grade in BRATS
    '''
    import glob
    from random import shuffle
    train = list()
    valid = list()
    test = list()
    
    for gp in groups_pattern:
        filenames = glob.glob(gp)
        shuffle(filenames)
        first_valid_sample = int(ratio_train*len(filenames))
        first_test_sample = first_valid_sample+int(ratio_valid*(len(filenames)-first_valid_sample))
              
        train += filenames[0:first_valid_sample]
        valid += filenames[first_valid_sample:first_test_sample]
        test += filenames[first_test_sample:]
    
    return train, valid, test


def rescale(x, xmin, xmax, a, b):
    ''' Rescales x from range (xmin, xmax) to an y in (a, b)'''
    return (a + (b-a)*(x-xmin)/(xmax-xmin))  

def pack_dataset(file_paths, dataset_name, dataset_suffix, dataset_min, dataset_max, center_crop=[180,180,128], normalize_to='mri', percentiles=[2, 98]):
    '''
    Preprocess the dataset and save a .tfrecords file.
    
    :param file_paths: list of directories containing the .mha files
    :param dataset_name: Must be one of 'brats2015-Train-all', 'BD2Decide-T1T2'
    :param dataset_min: Minimum value present in the entire dataset, it is written to each record. Can be found using mha_helpers.
    :param dataset_min: Maximum value present in the entire dataset, it is written to each record. Can be found using mha_helpers.
    :param center_crop: size [x, y, z] of the sub volume to center-crop from the 3d mri and labels
    :param normalize_to: can be None (no normalization), 'dataset', 'mri' [Default] - normalized according to all the 3d volume or 'slice'.
    :param percentiles: [low=2, hi=98] percentiles of the extreme values for normalization
    '''
    # Code adapted from https://www.tensorflow.org/tutorials/load_data/tf_records

    def get_one_sample():
        for f, mri_path in enumerate(file_paths):
            #### SPECIFY HERE HOW TO FETCH ALL THE MODALITIES CORRESPONDING TO THE SAME MRI SCAN, according to each dataset
            print("Parsing sample {} of {}".format(f, len(file_paths)))
            if 'brats2015' in dataset_name.lower() or 'bd2decide' in dataset_name.lower():
                mha_paths = glob.glob(mri_path+'*/*.mha') if 'brats2015' in dataset_name.lower() else glob.glob(mri_path+'/*.mha')
                # Preparing an intermediate record: {'path', 't1', 't1c', 't2', 'flair', 'gt', ...}

                # Finding the target given a filename
                modalities = {m['mri_type']: m for m in [preprocess_mha(mha, dataset_min, dataset_max, center_crop, normalize_to, percentiles) for mha in mha_paths]}
                slices = preprocess_slice(modalities, dataset_name)

                for slice_meta in slices:
                    # This code produces a feature dictionary for each entry in slice meta according to its type
                    feature_description = {}
                    # Serializing data
                    for key in slice_meta.keys():
                        feature_description[key] = serialize(slice_meta[key])

                    yield tf.train.Example(features=tf.train.Features(feature=feature_description))
                    #return tf.train.Example(features=tf.train.Features(feature=feature_description))           
                    
            else:
                raise NotImplementedError("Parser for this dataset is not found.")
            
            
    record_generator = get_one_sample()
    iscropped  = '_crop' if center_crop else ''
    isnorm = '' if normalize_to is None else '_'+normalize_to
    outpath = '../datasets/{}_{}{}{}.tfrecords'.format(dataset_name, dataset_suffix, iscropped, isnorm)

    options = tf.io.TFRecordOptions(compression_type="GZIP") if use_gzip_compression else None
    with tf.io.TFRecordWriter(outpath, options=options) as writer:
        print("Writing samples to {}...".format(outpath))
        for sample in record_generator:
            writer.write(sample.SerializeToString())
        print("Samples written to {}")


def load_dataset(name, mri_type, center_crop=None, random_crop=None, filter=None, batch_size=32, cache=True, prefetch_buffer=1, shuffle_buffer=128, interleave=1, cast_to=tf.float32, clip_labels_to=0.0, take_only=None, shuffle=True, infinite=False, n_threads=os.cpu_count()):
    '''
    Load a tensorflow dataset <name> (see definition in dataset_helpers).
    :param name: Name of the dataset
    :param mri_type: list of MRI sequencing to include in the dataset. Each modality will form a new channel in the resulting sample.
    :param filter: Lambda expression for filtering the data
    :param batch_size: batch size of the returned tensors
    :param cache: [True] wether to cache data in main memory
    :param buffer_size: Buffer size used to prefetch data
    :param interleave: If true, subsequent calls to the iterator will generate <interleave> equal batches. Eg. if 3, batch returned will be [A A A B B B ....]
    :param cast_to: [tf.float32] cast image data to this dtype
    :param only_nonempty_labels: [True] If true, filters all the samples that have a completely black (0.0) label map (segmentation)
    :param clip_labels_to: [0.0] If > 0, clips all the segmentation labels to the provided value. eg. providing a 1 yould produce a segmentation with only 0 and 1 values
    :param take_only: [None] If > 0, only returns <take_only> samples from the given dataset before starting a new iteration.
    :return:
    '''

    def parse_sample(sample_proto):
        parsed = tf.io.parse_single_example(sample_proto, get_feature_description(["OT"]+mri_type))
        # Decoding image arrays
        
        slice_shape = [parsed['OT_x_dimension'.format(mri_type[0])], parsed['OT_y_dimension'], 1]
        # Decoding the ground truth
        parsed['seg'] = tf.cast(tf.reshape(tf.io.decode_raw(parsed['OT_mri'], tf.float32), shape=slice_shape), dtype=cast_to)
        # Decode each channel and stack in a 3d volume
        stacked_mri = list()
        for mod in mri_type:
            stacked_mri.append(tf.cast(tf.reshape(tf.io.decode_raw(parsed['{}_mri'.format(mod)], tf.float32), shape=slice_shape), dtype=cast_to))
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
                cropped = tf.image.resize_with_crop_or_pad(mri_seg,center_crop[0],center_crop[1])
            # Splitting back
            parsed['mri'] = cropped[:,:,:len(mri_type)]
            parsed['seg'] = cropped[:,:,len(mri_type):]
        return parsed
    path = '../datasets/{}.tfrecords'.format(name)
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP' if use_gzip_compression else "")
    dataset = dataset.filter(filter) if filter is not None else dataset
    dataset = dataset.take(take_only) if take_only is not None else dataset
    
    if shuffle and infinite:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer))
    else:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True) if shuffle else dataset
        dataset = dataset.repeat() if infinite else dataset
    dataset = dataset.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size) if batch_size > 0 else dataset
    dataset = dataset.cache() if cache else dataset

    if interleave > 1:
        dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(interleave), cycle_length=n_threads, block_length=interleave, num_parallel_calls=n_threads)
        
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def find_extreme_values(file_paths, pattern, center_crop=None):
    '''
    Scan every given .mha file to find max and minimum voxel intsity values.
    Returns the full list of .mha files obtained combining the file paths and the given pattern, and the min/max of the whole dataset.
    '''

    mha_files = list()
    for f in file_paths:
        mha_files += glob.glob(f+pattern)
        
    max_v = 0
    min_v = 0

    for i, f in enumerate(mha_files):
        itkimage = sitk.GetArrayFromImage(sitk.ReadImage(f)).transpose((0, 2, 1))
        if center_crop:
            itkimage = itkimage[int(itkimage.shape[0]/2 - center_crop[2]/2):int(itkimage.shape[0]/2 + center_crop[2]/2), int(itkimage.shape[1]/2 - center_crop[0]/2):int(itkimage.shape[1]/2 + center_crop[0]/2),int(itkimage.shape[2]/2 - center_crop[1]/2):int(itkimage.shape[2]/2 + center_crop[1]/2)]
        if itkimage.max == 4:
            # This is a label, skip
            continue
   
        max_v = max(itkimage.max(), max_v)
        min_v = min(itkimage.min(), min_v)

    return min_v, max_v
  

def prepare_brats():
    name = 'brats2015-Train-all'
    # Pipeline for preparing BRATS2015 for SegAN
    center_crop = [180,180,128]
    brats_train, brats_valid, brats_test = split_train_validation_test(['../../datasets/BRATS2015/BRATS2015_Training/HGG/*/', '../../datasets/BRATS2015/BRATS2015_Training/LGG/*/'], ratio_train=0.8, ratio_valid=0.5)
    train_min, train_max = find_extreme_values(file_paths=brats_train, pattern='/*/*.mha', center_crop=center_crop)
    valid_min, valid_max = find_extreme_values(file_paths=brats_valid, pattern='/*/*.mha', center_crop=center_crop)
    test_min, test_max = find_extreme_values(file_paths=brats_test, pattern='/*/*.mha', center_crop=center_crop)
    pack_dataset(brats_train, 'brats2015', dataset_suffix='training', dataset_min=train_min, dataset_max=train_max, center_crop=center_crop, normalize_to='mri')
    pack_dataset(brats_valid, 'brats2015', dataset_suffix='validation', dataset_min=valid_min, dataset_max=valid_max, center_crop=center_crop, normalize_to='mri')
    pack_dataset(brats_test, 'brats2015', dataset_suffix='testing', dataset_min=test_min, dataset_max=test_max, center_crop=center_crop, normalize_to='mri')

def prepare_bd2decide():
    name = 'bd2decide'
    # Pipeline for preparing BD2Decide for SegAN
    center_crop = [180,180,128]
    bd2_train, bd2_valid, bd2_test = split_train_validation_test(['../../datasets/BD2Decide/resampled/T1T2_reshaped/*'], ratio_train=0.8, ratio_valid=0.5)
    train_min, train_max = find_extreme_values(file_paths=bd2_train, pattern='/*.mha', center_crop=center_crop)
    valid_min, valid_max = find_extreme_values(file_paths=bd2_valid, pattern='/*.mha', center_crop=center_crop)
    test_min, test_max = find_extreme_values(file_paths=bd2_test, pattern='/*.mha', center_crop=center_crop)
    
    pack_dataset(bd2_train, 'BD2Decide-T1T2', dataset_suffix='training', dataset_min=train_min, dataset_max=train_max, center_crop=center_crop, normalize_to='mri')
    pack_dataset(bd2_valid, 'BD2Decide-T1T2', dataset_suffix='validation', dataset_min=valid_min, dataset_max=valid_max, center_crop=center_crop, normalize_to='mri')
    pack_dataset(bd2_test, 'BD2Decide-T1T2', dataset_suffix='testing', dataset_min=test_min, dataset_max=test_max, center_crop=center_crop, normalize_to='mri')
