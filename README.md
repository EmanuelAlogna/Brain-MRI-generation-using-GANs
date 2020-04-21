# Politecnico di Milano - Thesis Repository

### Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)


# In Details

- The dataset used in this work is [BRATS2015](https://www.smir.ch/BraTS/Start2015) (preprocessed using the file *write_tfrecord_script.py*).


- To validate the predictions using the segmentation model, it's necessary to first write a tf.record that contains all generated samples (use  *[generate_images_to_segment.ipynb](generation/generate_images_to_segment.ipynb)*) and then test quantitatively (use *[dsc_from_generated_samples.ipynb](segmentation/dsc_from_generated_samples.ipynb)*) or qualitatively (use *[segmentations_from_generated_samples.ipynb](segmentation/segmentations_from_generated_samples.ipynb)*) the segmentations.

Folder structure
--------------

```
├──  data                       - this folder contains a batch of T1c predictions from our models. 
│   ├── batch_from_MIGAN_t1c.npy 
│   ├── batch_from_MIpix2pix_t1c.npy 
│   └── batch_from_pix2pix_t1c.npy
│ 
│
│
├──  documentation/thesis
│   └── b2020_04_Alogna.pdf     - this file contains the master thesis pdf.
│ 
│
│
├── generation        - this folder contains any source of code related to the MRI generation.
│   ├── experiments   - this folder contains the experiments conducted: Skip and Internal connection analyses
│   ├── models        - this folder contains the code needed to train the 14 models of our work. 
│   ├── tests         - this folder contains the notebooks to test, qualitatively and quantitavely, the models.
│   │
│   ├── compute_baselines.ipynb            - this file contains the code to compute baselines score between modalities.
│   ├── dataset_helpers.py                 - this script contains the code to read tf.record.
│   ├── evaluation_metrics.py              - this script contains the implemented evaluation metrics.
│   ├── generate_images_to_segment.ipynb   - this file stores all the predictions inside a tf.record.
│   └── write_tfrecord_script.py           - this script is needed to convert .mha files in tf.record.
│
│
│
├── images      - this folder contains images used in the documentation.
│
│
│
├── segmentation      - this folder contains all the files needed to segment the predictions
│   ├── models        - this file contains the models implemented and trained by [1].
│   │
│   ├── DeepMRI.py                                  - script from [1].
│   ├── SegAN_IO_arch.py                            - script from [1].
│   ├── SeganCATonColab.ipynb                       - this file allows to test the predictions contained in data folder
│   ├── dataset_helpers.py                          - script with some utility functions.
│   ├── dsc_from_generated_samples.ipynb            - this file computes DSC from the segmentation of the predictions.
│   └── segmentations_from_generated_samples.ipynb  - this file shows the qualitative results of segmentation of the predictions.

```
Models overview
--------------

- pix2pix was developed following the [official implementation of Tensorflow](http://tensorflow.org/tutorials/generative/pix2pix) and the original paper ([Isola et al., 2016](
https://arxiv.org/abs/1611.07004)).

- MI-pixpix is a modified version of pix2pix, adapted to the multi-input scenario. 

- MI-GAN is a modified version of the MM-GAN ([Sharma et al., 2019](https://arxiv.org/abs/1904.12200)), adapted to the many-to-one scenario.


Generated MRI samples
--------------
<br />

<img src="/images/generated_samples.png" alt="alt text" width="720" height="">

Segmentations using GANs predictions
--------------

<br />
<img src="/images/segmented_samples.png" alt="alt text" width="720" height="">

References
--------------

[1] [E. Giacomello, D. Loiacono, and L. Mainardi, “Brain mri tumor segmentation with adversarial networks,” 2019](https://arxiv.org/abs/1910.02717)
