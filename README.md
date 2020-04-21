# Brain-MRI-generation-using-GANs
Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis


# In Details

Project architecture 
--------------

Folder structure
--------------

```
├──  documentation/thesis
│   ├── b2020_04_Alogna.pdf     - this file contains the master thesis pdf.
│ 
│
│
├── generation      - this folder contains any source of code related to the MRI generation.
│   └── experiments
│   └── models                       - this folder contains the 14 models trained. 
│   └── tests                        - this folder contains the notebooks to test, qualitatively and quantitavely, the models.
│   └── compute_baselines.ipynb      - this file contains the abstract class of the trainer.
│   └── dataset_helpers.py           - this file contains the abstract class of the trainer.
│   └── evaluation_metrics.py        - this file contains the abstract class of the trainer.
│   └── egenerate_images_to_segment.ipynb     - this file contains the abstract class of the trainer. 
│   └── write_tfrecord_script.py    - this file contains the abstract class of the trainer.
│
│
│
├── segmentation      - this folder contains all the files needed to segment the predictions
│   └── models  - this file contains the abstract class of the trainer.
│   └── DeepMRI.py  - this file contains the abstract class of the trainer.
│   └── DSegAN_IO_arch.py - this file contains the abstract class of the trainer.
│   └── DSeganCATonColab.ipynb  - this file contains the abstract class of the trainer.
│   └── Ddataset_helpers.py - this file contains the abstract class of the trainer.
│   └── Ddsc_from_generated_samples.ipynb - this file contains the abstract class of the trainer.
│   └── Dsegmentations_from_generated_samples.ipynb - this file contains the abstract class of the trainer.
