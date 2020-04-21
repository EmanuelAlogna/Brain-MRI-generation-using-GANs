
# Some Details

- All the experiments are conducted using model synthesizing the missing modality T2flair.
- All the experiments contain mainly two parts: one used to evaluate quantitatively the performances of the models and one to plot the predictions.
- The function *restore_models* allows to retrieve the model trained (that can be pix2pix/MI-pix2pix/MI-GAN) and then initialize all the generators variants, each one with a different number of connections (skips or internal) turned off / perturbed.

Folder structure
--------------

```
├──  experiments               
│   ├── internal_connections_analysis.ipynb   - in this experiment we turned off the internal connections
│   ├── skip_connections_analysis_1.ipynb     - in this experiment we turned off the skip connections
│   └── skip_connections_analysis_2.ipynb     - in this experiment we perturbing the skip connections

```
