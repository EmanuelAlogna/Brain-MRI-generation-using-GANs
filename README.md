# BrainMRI-generation-using-GANs
Brain Magnetic Resonance Imaging Generation using Generative Adversarial Networks - master thesis

#################################

Abstract of the work

Magnetic Resonance Imaging (MRI) is nowadays one of the most common medical imaging techniques, due to the non-invasive nature of this type of scan that can acquire many modalities (or sequences), each one with a different image appearance and unique insights about a particular disease. However, it is not always possible to obtain all the sequences required, due to many issues such as prohibitive scan times or allergies to contrast agents. To overcome this problem and thanks to the recent improvements in Deep Learning, in the last few years researchers have been studying the applicability of Generative Adversarial Network (GAN) to synthesize the missing modalities. Our work proposes a detailed study that aims to demonstrate the power of GANs in generating realistic MRI scans of brain tumors through the implementation of different models. We trained in particular two kind of networks which differ from the number of sequences received in input, using a dataset composed by 274 different volumes from subjects with brain tumor, and, among a set of different evaluation metrics implemented to test our results, we validated the quality of the predicted images using also a segmentation model. In addition, we analysed the GANs trained by performing some experiments to understand how the information passes through the generator network. Our results show that the synthesized sequences are highly accurate, realistic and in some cases indistinguishable from true brain slices of the dataset, highlighting the advantage of multi-modal models that, compared to the unimodal ones, can exploit the correlation between available sequences. Moreover, they demonstrate the effectiveness of skip connections and their crucial role in the generative process by showing the significant degradation in the performances, analysed in both a qualitative and quantitative way, when these channels are turned off or perturbed.


#################################
