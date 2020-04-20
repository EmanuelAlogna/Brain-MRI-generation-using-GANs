import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import dataset_helpers as dh
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Setting allow_growth for gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model running on CPU")


class DeepMRI():
    def __init__(self, batch_size, size, mri_channels, output_labels=1, model_name='DeepMRI', max_epochs=100000):
        self.batch_size = batch_size
        self.size = size
        self.output_labels = output_labels
        self.mri_shape = (size, size, mri_channels)
        if output_labels == 1:
            self.label_shape = (size, size, output_labels)
        else:
            self.label_shape = (size, size, output_labels + 1)
        self.train_dataset = None
        self.max_epochs = max_epochs
        self.ref_sample = None
        self.model_name = model_name
        self.save_path = 'models/{}/'.format(model_name)
        self.current_epoch = 0
        self.save_every = 1 # If you change this and load a saved model, epoch count will be wrong. Update current_epoch before starting the train.
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.metrics_names = ['sensitivity','specificity','false_positive_rate','precision','dice_score','balanced_accuracy', 'true_positives', 'false_positives', 'false_negatives', 'true_negatives']
        self.log_column_names = [met+'_'+str(lab) for met in self.metrics_names for lab in range(self.label_shape[-1])]
        
    def build_model(self, load_model='last', transfer=False, seed=1234567890, arch=None, g_opt=None, d_opt=None):
        ''' If load_model is 'last' load the most recent checkpoint (creates a new one if none are found), otherwise loads the specified one.
        '''
        if seed is not None:
            tf.random.set_seed(seed)
        
        if arch is None:
            import SegAN_IO_arch as arch
        
        print("Using architecture: {}".format(arch.__name__))
        self.arch = arch
        self.generator = arch.build_segmentor(self.mri_shape, seg_channels=self.label_shape[-1])
        self.discriminator = arch.build_critic(self.mri_shape, self.label_shape)
        self.g_optimizer = tf.optimizers.RMSprop(learning_rate=0.00002) if g_opt is None else g_opt
        self.d_optimizer = tf.optimizers.RMSprop(learning_rate=0.00002) if d_opt is None else d_opt
        
        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator, g_optimizer=self.g_optimizer, d_optimizer=self.g_optimizer)
        last_ckpt = tf.train.latest_checkpoint(self.save_path)
        
        self.tb_path_training = self.save_path+'tensorboard_log/training/'
        self.tb_path_validation = self.save_path+'tensorboard_log/validation/'
        os.makedirs(self.tb_path_training, exist_ok=True)
        os.makedirs(self.tb_path_validation, exist_ok=True)
        
        self.tb_writer_t = tf.summary.create_file_writer(self.tb_path_training)
        self.tb_writer_v = tf.summary.create_file_writer(self.tb_path_validation)
        self.log_train_path = self.save_path + 'log_train.csv'
        self.log_valid_path = self.save_path + 'log_valid.csv'
        
        
        
        #self.train_metrics = Logger(self.save_path+'log_train.csv')
        #self.valid_metrics = Logger(self.save_path+'log_valid.csv')
        
        if load_model=='last' and last_ckpt is not None:
            print("Latest Checkpoint is: {}".format(last_ckpt))
            self.ckpt.restore(last_ckpt)
            self.current_epoch = int(last_ckpt.split('-')[0].split('_')[-1]) + 1
            print("Loaded model from: {}, next epoch: {}".format(load_model, self.current_epoch))
        else:
            if load_model != 'last':
                print("Loading", load_model)
                self.ckpt.restore(load_model)
                if transfer:
                    self.current_epoch = 0
                    print("Transfering model from: {}, next epoch: {}".format(load_model, self.current_epoch))
                else:
                    self.current_epoch = int(load_model.split('/')[-1].split('-')[0].split('_')[-1]) + 1
                    print("Resuming model from: {}, next epoch: {}".format(load_model, self.current_epoch))
            else:
                print("Created new model")
        
        if os.path.isfile(self.log_train_path):
            self.log_train = pd.read_csv(self.log_train_path)
            self.log_train = self.log_train[self.log_train['epoch']<self.current_epoch]
        else:
            self.log_train = pd.DataFrame()
        if os.path.isfile(self.log_valid_path):
            self.log_valid = pd.read_csv(self.log_valid_path)
            self.log_valid = self.log_valid[self.log_valid['epoch']<self.current_epoch]
        else:
            self.log_valid = pd.DataFrame()
    
    def set_training_dataset(self, dataset):
        if self.train_dataset is not None:
            print("Unloading previous dataset")
            del self.train_dataset
        self.train_dataset = dataset
    
    def set_validation_dataset(self, dataset):
        if self.validation_dataset is not None:
            print("Unloading previous dataset")
            del self.validation_dataset
        self.validation_dataset = dataset
    
    def set_testing_dataset(self, dataset):
        if self.test_dataset is not None:
            print("Unloading previous dataset")
            del self.test_dataset
        self.test_dataset = dataset
   
    def load_dataset(self, dataset, mri_types, training_shuffle=True, training_random_crop=True, training_center_crop=False, testval_center_crop=True):
        ''' 
        Load the given datasets. 
        :param dataset - A dict containing at least the keys 'training' and 'validation' ('testing' is optional). The values are the filenames of .tfrecords file for the given dataset (relative to ../datasets/, without .tfrecords extension)
        :param mri_types - A list of MRI Modalities corresponding to the network input channel, as specified in the dataset feature columns.
        
        '''
        self.mri_types = mri_types
        
        training_random_crop = list(self.mri_shape) if training_random_crop == True else None
        training_center_crop = list(self.mri_shape) if training_center_crop == True else None
        testval_center_crop = list(self.mri_shape) if testval_center_crop == True else None
        
        if any([d is not None for d in [self.train_dataset, self.validation_dataset, self.test_dataset]]):
            print("Unloading previous dataset")
            del self.train_dataset
            del self.validation_dataset
            del self.test_dataset
            
        
        print("Loading training dataset {} with modalities {}".format(dataset['training'], ','.join(mri_types)))
        self.train_dataset = lambda: dh.load_dataset(dataset['training'],
                                mri_type=mri_types,
                                ground_truth_column_name='seg' if 'brats2019' in dataset['training'] else "OT",
                                clip_labels_to=self.output_labels,
                                random_crop=training_random_crop,
                                center_crop=training_center_crop,                 
                                batch_size=self.batch_size,
                                prefetch_buffer=1,
                                infinite=False, 
                                cache=False,
                                shuffle=training_shuffle
                                )
        print("Loading validation dataset {} with modalities {}".format(dataset['validation'], ','.join(mri_types)))
        self.validation_dataset = lambda: dh.load_dataset(dataset['validation'],
                                        mri_type=mri_types,
                                        ground_truth_column_name='seg' if 'brats2019' in dataset['validation'] else "OT",
                                        clip_labels_to=self.output_labels,
                                        center_crop=testval_center_crop,
                                        batch_size=self.batch_size,
                                        prefetch_buffer=1,
                                        infinite=False, 
                                        cache=False,
                                        shuffle=False
                                        )
        if 'testing' in dataset:
            print("Loading testing dataset {} with modalities {}".format(dataset['testing'], ','.join(mri_types)))
            self.test_dataset = lambda: dh.load_dataset(dataset['testing'],
                                            mri_type=mri_types,
                                            ground_truth_column_name='seg' if 'brats2019' in dataset['testing'] else "OT",
                                            clip_labels_to=self.output_labels,
                                            center_crop=testval_center_crop,
                                            batch_size=self.batch_size,
                                            prefetch_buffer=1,
                                            infinite=False, 
                                            cache=False,
                                            shuffle=False
                                                )

        self.train_dataset_length = None
        self.validation_dataset_length = None
        self.test_dataset_length = None
        
        # Selecting one random sample from validation set as reference for the predictions
        for row in self.validation_dataset():
            if row['seg'].numpy()[0].any():
                self.ref_sample = row['mri'].numpy()[0], row['seg'].numpy()[0]
                print("Done.")
                break
        
    
    
    @tf.function
    def compute_metrics(self, y_true, y_pred, threshold=0.5):
        # Boolean segmentation outputs
        # Condition Positive - real positive cases
        CP = tf.greater(y_true, threshold)  # Ground truth
        # Predicted Condition Positive - predicted positive cases
        PCP = tf.greater(y_pred, threshold)   # Segmentation from S (prediction)
        # Codition Negative
        CN = tf.math.logical_not(CP)
        # Predicted Condition Negative
        PCN = tf.math.logical_not(PCP)

        TP = tf.math.count_nonzero(tf.math.logical_and(CP, PCP), axis=(1, 2))
        FP = tf.math.count_nonzero(tf.math.logical_and(CN, PCP), axis=(1, 2))
        FN = tf.math.count_nonzero(tf.math.logical_and(CP, PCN), axis=(1, 2))
        TN = tf.math.count_nonzero(tf.math.logical_and(CN, PCN), axis=(1, 2))

        # TPR/Recall/Sensitivity/HitRate, Probability of detection
        sensitivity = tf.where(tf.greater(TP+FN, 0), TP/(TP+FN), 1.0)
        # TNR/Specificity/Selectivity, Probability of false alarm
        specificity = tf.where(tf.greater(TN+FP, 0), TN/(TN+FP), 1.0)
        # False Positive Rate / fall-out
        false_positive_rate = 1 - specificity
        # Precision/ Positive predictive value
        precision = tf.where(tf.greater(TP+FP, 0), TP/(TP+FP), 1.0)
        # Dice score (Equivalent to F1-Score)
        dice_score = tf.where(tf.greater(TP+FP+FN, 0), (2*TP)/(2*TP+FP+FN), 1.0)
        # (Balanced) Accuracy - Works with imbalanced datasets
        balanced_accuracy = (sensitivity + specificity)/2.0
                
        # When editing this also edit the Logger class accordingly
        return [sensitivity, specificity, false_positive_rate, precision, dice_score, balanced_accuracy,  TP, FP, FN, TN]
        
        
    @tf.function
    def train_step(self, x, y, train_g=True, train_d=True):
        '''
        Performs a training step.
        :param x: batch of training data
        :param y: batch of target data
        :train_g: A tf.constant (Bool) telling if g has to be trained [True]
        :train_d: A tf.constant (Bool) telling if d has to be trained [True]
        '''
        # FIXME: Here Pix2Pix example uses 2 tapes
        #with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        with tf.GradientTape(persistent=True) as tape:
            g_output = self.generator(x, training=train_g)
            d_real = self.discriminator([x, y], training=train_d)
            d_fake = self.discriminator([x, g_output], training=train_d)
            
            g_loss = self.arch.loss_g(d_real, d_fake, g_output, y)
            d_loss = self.arch.loss_d(d_real, d_fake)
        if train_g == True:
            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        if train_d == True:
            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        del tape

        return [g_loss, d_loss], self.compute_metrics(y, g_output)
    
    @tf.function
    def validation_step(self, x, y):
        g_output = self.generator(x, training=False)
        d_real = self.discriminator([x, y], training=False)
        d_fake = self.discriminator([x, g_output], training=False)
        g_loss = self.arch.loss_g(d_real, d_fake, g_output, y)
        d_loss = self.arch.loss_d(d_real, d_fake)
        return [g_loss, d_loss], self.compute_metrics(y, g_output)
    
    def log_step(self, log, row, losses, metrics):
        stacked = np.stack([met.numpy().squeeze().astype(np.float32) for met in metrics], axis=1)
        if self.output_labels > 1:
            reshaped = np.reshape(stacked, (stacked.shape[0], stacked.shape[1]*stacked.shape[2]))
        else:
            reshaped = stacked
        step_log = pd.DataFrame(reshaped, columns=self.log_column_names)
        for t in self.mri_types:
            step_log[t+'_path'] = [s.decode('utf-8') for s in row[t+'_path'].numpy()]
        if losses is not None:
            step_log['loss_g'] = losses[0].numpy()
            step_log['loss_d'] = losses[1].numpy()
        step_log['learning_rate_g'] = self.g_optimizer.lr.numpy()
        step_log['learning_rate_d'] = self.d_optimizer.lr.numpy()
        log = log.append(step_log, ignore_index=True)
        return log
            
    
    def log_epoch(self, slice_log, run, epoch_number, tracked_metric, tracked_metric_maximize=True, save_every_epochs=20):
        ''' Calculates the metrics for each 3D volume for the full training or validation run. 
        Writes the tensorboard writer and saves the best model on validation
        :param slice_log - DataFrame containing the metrics for each slice processed in the current epoch
        :param run - "training" or "validation", for writing to the corresponding logs
        :param epoch_number - current epoch number, written on the log and on tensorboard graph
        :param tracked_metric - Metric to track for saving the best model. If None, the model is saved at a regular interval governed by save_every_epoch
        :param tracked_metric_maximize - Whether if the given tracked_metric is to be maximized or minimized. [Default: True]
        :param save_every_epochs - Number of epochs between two save() calls. Has effect only if tracked_metric is None.'''
        
        if run == 'training':
            epoch_log = self.log_train
            tb_writer = self.tb_writer_t
        if run == 'validation':
            epoch_log = self.log_valid
            tb_writer = self.tb_writer_v
            
        mri_groups = slice_log.groupby(['{}_path'.format(t) for t in self.mri_types])
        sums = mri_groups.sum()
        means = mri_groups.mean()
        
        per_mri_stats = pd.DataFrame()
        for t in range(self.label_shape[-1]):
            per_mri_stats['dice_score_{}'.format(t)] = 2.0*sums['true_positives_{}'.format(t)]/(2.0*sums['true_positives_{}'.format(t)] + sums['false_positives_{}'.format(t)] + sums['false_negatives_{}'.format(t)])
            per_mri_stats['precision_{}'.format(t)] = sums['true_positives_{}'.format(t)]/(sums['true_positives_{}'.format(t)] + sums['false_positives_{}'.format(t)])
            per_mri_stats['sensitivity_{}'.format(t)] = sums['true_positives_{}'.format(t)]/(sums['true_positives_{}'.format(t)] + sums['false_negatives_{}'.format(t)])
        per_mri_stats['loss_g'] = means['loss_g']
        per_mri_stats['loss_d'] = means['loss_d']
        per_mri_stats['learning_rate_g'] = means['learning_rate_g']
        per_mri_stats['learning_rate_d'] = means['learning_rate_d']
        
        # METRICS FOR BRATS2015 CHALLENGE
        definitions = (('complete_tumor_2019', [1,2,4]), ('tumor_core_2019', [1,3,4]), ('complete_tumor', [1,2,3,4]), ('tumor_core', [1,3,4]), ('enhancing_tumor', [4]))
        for chal_met, labs in definitions:
            tp = sums.loc[:, sums.columns.isin(['true_positives_{}'.format(l) for l in labs])].sum(axis=1)
            tn = sums.loc[:, sums.columns.isin(['true_negatives_{}'.format(l) for l in labs])].sum(axis=1)
            fp = sums.loc[:, sums.columns.isin(['false_positives_{}'.format(l) for l in labs])].sum(axis=1)
            fn = sums.loc[:, sums.columns.isin(['false_negatives_{}'.format(l) for l in labs])].sum(axis=1)
            per_mri_stats['dice_score_{}'.format(chal_met)] = 2.0*tp/(2*tp + fp + fn)
            per_mri_stats['precision_{}'.format(chal_met)] = tp/(tp + fp)
            per_mri_stats['sensitivity_{}'.format(chal_met)] = tp/(tp + fn)

        current_epoch_log = per_mri_stats.mean().to_frame().transpose()
        current_epoch_log['epoch'] = epoch_number
        
        # Visualization and saving
        if run in ['training', 'validation']:
            with tb_writer.as_default():
                for met in current_epoch_log.columns:
                    if met == 'epoch':
                        continue
                    tf.summary.scalar(met, current_epoch_log[met].item(), step=epoch_number)
                tb_writer.flush()


            if run == 'validation':
                if tracked_metric is not None:
                    if len(epoch_log)==0 or \
                       (tracked_metric_maximize and current_epoch_log[tracked_metric].item() >= epoch_log[tracked_metric].max()) or \
                       (not tracked_metric_maximize and current_epoch_log[tracked_metric].item() <= epoch_log[tracked_metric].min()):
                            print("Found new best model for {}, saving...".format(tracked_metric))
                            self.ckpt.save(self.save_path+"best_{}_{}".format(tracked_metric, epoch_number))
                            #self.log_prediction(epoch_number)
                else:
                    if epoch_number % save_every_epochs == 1:
                        self.ckpt.save(self.save_path+"last_epoch_{}".format(epoch_number))
                        #self.log_prediction(epoch_number)
    
        if run == 'training':
            self.log_train = self.log_train.append(current_epoch_log, ignore_index=True)
            self.log_train.to_csv(self.log_train_path)
        if run == 'validation':
            self.log_valid = self.log_valid.append(current_epoch_log, ignore_index=True)
            self.log_valid.to_csv(self.log_valid_path)
        if run == 'testing':
            return current_epoch_log

    
    def alternated_training(self, n_gen, n_disc, start_with='d'):
        '''
        Iterator returning a tuple (train_g, train_d) of Bools indicating if g or d has to be trained this step, in an alternated fashion.
        :param n_gen: how many times g has to be trained before training d        
        :param n_disc: how many times d has to be trained before training g
        :start_with: String, can be either 'g' or 'd'.
        '''        
        switch = start_with
        c = 0
        tg, td = True, True
        while True:
            if switch=='d':
                tg, td = False, True
                c += 1
                if c >= n_disc:
                    switch = 'g'
                    c = 0
            elif switch=='g':
                tg, td = True, False
                c += 1
                if c >= n_gen:
                    switch = 'd'
                    c = 0
            yield tg, td

    
    def train(self, alternating_steps=None, tracked_metric='dice_score', tracked_metric_maximize=True, save_every_epochs=20, max_epochs=None):
        ''' 
            Train the network, saving metrics every epoch and saving the best model on the tracked metric.
            Supports alternating training. 
            :param alternating_steps: (steps_g, steps_d) or None. How many steps each network has to be trained before starting training the other. \
            If None, both network are trained each step on the same batch of data (train happens independently on G and D in any case).
            Ie. None: G is trained on X1, then D on X1, G on X2, D on X2...
            If (1, 1), G is trained on X1, then D on X2, G on X3, D on X4...
            
        '''

        net_switch = self.alternated_training(alternating_steps[0], alternating_steps[1]) if alternating_steps is not None else None
               
        for e in range(self.current_epoch, self.max_epochs):
            # Logs that store metrics for each slice in the current epoch
            training_logger = pd.DataFrame() 
            validation_logger = pd.DataFrame()
                        
            self.train_progress = tk.utils.Progbar(self.train_dataset_length, stateful_metrics=['loss_g', 'loss_d'] + self.metrics_names)
            # Training Step
            for i, row in enumerate(self.train_dataset()):
                # Alternated training
                if net_switch is None or (e == self.current_epoch and i==0):
                    # The fist step we train both g and d for initializing the needed tensors
                    train_g, train_d = True, True
                else:
                    train_g, train_d = next(net_switch)
                    
                # Train step and metric logging
                train_losses, train_metrics = self.train_step(row['mri'], row['seg'], train_g=train_g, train_d=train_d)
                training_logger = self.log_step(training_logger, row, train_losses, train_metrics)
                
                self.train_progress.update(i, (('loss_g', train_losses[0]) , ('loss_d', train_losses[1])))
            self.train_dataset_length = i + 1
            self.log_epoch(training_logger, 'training', e, tracked_metric, tracked_metric_maximize, save_every_epochs)
            
            
            # Validation Step
            self.valid_progress = tk.utils.Progbar(self.validation_dataset_length, stateful_metrics=self.metrics_names)
            for i, row in enumerate(self.validation_dataset()):
                valid_losses, valid_metrics = self.validation_step(row['mri'], row['seg'])
                validation_logger = self.log_step(validation_logger, row, valid_losses, valid_metrics)
                self.valid_progress.update(i, (('loss_g', valid_losses[0]) , ('loss_d', valid_losses[1])))
            self.validation_dataset_length = i + 1
            # Log validation epoch (and save if necessary)
            self.log_epoch(validation_logger, 'validation', e, tracked_metric, tracked_metric_maximize, save_every_epochs)
            
            if max_epochs is not None:
                if e > max_epochs:
                    print("Max epochs reached, terminating...")
                    break
            
               
    def evaluate(self, dataset='testing'):
        ''' 
        Evaluate the laoded model on the given dataset ('validation',  'testing' or 'merged'). Dataset must be loaded beforehand.
        :param csv_path: csv to save the results 
        '''
        assert dataset in ['validation', 'testing', 'merged']
        if dataset == 'validation':
            eval_dataset = [self.validation_dataset]
        if dataset == 'testing':
            eval_dataset = [self.test_dataset]
        if dataset == 'merged':
            eval_dataset = [self.validation_dataset, self.test_dataset]
        
        eval_logger = pd.DataFrame()
        self.eval_progress = tk.utils.Progbar(None, stateful_metrics=self.metrics_names)
        # Evaluation
        for dataset_split in eval_dataset:
            for i, row in enumerate(dataset_split()):
                eval_losses, eval_metrics = self.validation_step(row['mri'], row['seg'])
                eval_logger = self.log_step(eval_logger, row, eval_losses, eval_metrics)
                self.eval_progress.update(i, (('loss_g', eval_losses[0]) , ('loss_d', eval_losses[1])))
        
        # Log validation epoch (and save if necessary)
        return self.log_epoch(eval_logger, 'testing', 0, None)
  