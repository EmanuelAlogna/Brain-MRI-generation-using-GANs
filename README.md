# Brain-MRI-generation-using-GANs
Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis


# In Details

Project architecture 
--------------

<div align="center">

<img align="center" hight="600" width="600" src="https://github.com/Mrgemy95/Tensorflow-Project-Templete/blob/master/figures/diagram.png?raw=true">

</div>


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains any model of your project.
│   └── example_model.py
│
│
├── trainer             - this folder contains trainers of your project.
│   └── example_trainer.py
│   
├──  mains              - here's the main(s) of your project (you may need more than one main).
│    └── example_main.py  - here's an example of main that is responsible for the whole pipeline.

│  
├──  data _loader  
│    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│ 
└── utils
     ├── logger.py
     └── any_other_utils_you_need

```
