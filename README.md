# Politecnico di Milano - Thesis Repository

### Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)


# In Details

Project architecture 
--------------

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
│   ├── models        - this folder contains the 14 models trained. 
│   ├── tests         - this folder contains the notebooks to test, qualitatively and quantitavely, the models.
│   │
│   ├── compute_baselines.ipynb             - this file contains the abstract class of the trainer.
│   ├── dataset_helpers.py                  - this file contains the abstract class of the trainer.
│   ├── evaluation_metrics.py               - this file contains the abstract class of the trainer.
│   ├── generate_images_to_segment.ipynb   - this file contains the abstract class of the trainer. 
│   └── write_tfrecord_script.py            - this script is needed to .mha files and convert them in a tf.record.
│
│
│
├── segmentation      - this folder contains all the files needed to segment the predictions
│   ├── models        - this file contains the models implemented and trained by [1].
│   │
│   ├── DeepMRI.py                                  - script from [1].
│   ├── SegAN_IO_arch.py                            - script from [1].
│   ├── SeganCATonColab.ipynb                       - notebook from [1].
│   ├── dataset_helpers.py                          - script that allows to read tf.record.
│   ├── dsc_from_generated_samples.ipynb            - this file contains the abstract class of the trainer.
│   └── segmentations_from_generated_samples.ipynb  - this file contains the abstract class of the trainer.
