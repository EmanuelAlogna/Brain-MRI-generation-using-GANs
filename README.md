# Brain-MRI-generation-using-GANs
Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis


# In Details

Project architecture 
--------------

Folder structure
--------------

```
├──  documentation/thesis
│   ├── b2020_04_Alogna.pdf  - this file contains the abstract class of the model.
│ 
│
│
├── generation               - this folder contains any model of your project.
│   └── experiments
│   └── models  - this file contains the abstract class of the trainer.
│   └── tests  - this file contains the abstract class of the trainer.
│   └── compute_baselines.ipynb  - this file contains the abstract class of the trainer.
│   └── dataset_helpers.py  - this file contains the abstract class of the trainer.
│   └── evaluation_metrics.py  - this file contains the abstract class of the trainer.
│   └── egenerate_images_to_segment.ipynb - this file contains the abstract class of the trainer. 
│   └── write_tfrecord_script.py - this file contains the abstract class of the trainer.
│
│
│
├── segmentation             - this folder contains trainers of your project.
│   └── models  - this file contains the abstract class of the trainer.
│   └── DeepMRI.py  - this file contains the abstract class of the trainer.
│   └── DSegAN_IO_arch.py - this file contains the abstract class of the trainer.
│   └── DSeganCATonColab.ipynb  - this file contains the abstract class of the trainer.
│   └── Ddataset_helpers.py - this file contains the abstract class of the trainer.
│   └── Ddsc_from_generated_samples.ipynb - this file contains the abstract class of the trainer.
│   └── Dsegmentations_from_generated_samples.ipynb - this file contains the abstract class of the trainer.
