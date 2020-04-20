import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as layers
from tensorflow.keras import Model



def build_segmentor(mri_shape, seg_channels=1, clip=False):
    
    def S_enc_block(_input, f, k=4, strides=2, batchnorm=True, name='S_enc'):
        shape = _input.shape
        initializer = tk.initializers.RandomNormal(stddev=tf.math.sqrt(2./(k*k*f)))
        out = layers.Conv2D(f, k, strides=(strides, strides), use_bias=not batchnorm, kernel_initializer=initializer, padding='same', name=name+'_conv')(_input) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        out = layers.LeakyReLU(name=name+'_out')(out)
        return out
    
    def S_dec_block(_input, f, k=3, strides=1, batchnorm=True, name='S_dec', relu=True):
        out = layers.UpSampling2D(interpolation='bilinear')(_input)
        shape = out.shape
        initializer = tk.initializers.RandomNormal(stddev=tf.math.sqrt(2./(k*k*f)))
        out = layers.Conv2D(f, k, strides=(strides, strides), use_bias=not batchnorm, kernel_initializer=initializer, padding='same', name=name+'_conv')(out) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        if relu:
            out = layers.ReLU(name=name+'_out')(out)
        return out
    
    
    s_mri_in = layers.Input(mri_shape)
    
    enc1 = S_enc_block(s_mri_in, f=64, name='S_enc_1', batchnorm=False)
    enc2 = S_enc_block(enc1, f=128, name='S_enc_2')
    enc3 = S_enc_block(enc2, f=256, name='S_enc_3')
    enc4 = S_enc_block(enc3, f=512, name='S_enc_4')
    
    dec3 = S_dec_block(enc4, f=256, name='S_dec_3')
    skip3 = layers.Add(name='S_skip3')([dec3, enc3])
    dec2 = S_dec_block(skip3, f=128, name='S_dec_2')
    skip2 = layers.Add(name='S_skip2')([dec2, enc2])
    dec1 = S_dec_block(skip2, f=64, name='S_dec_1')
    skip1 = layers.Add(name='S_skip1')([dec1, enc1])
    out_unb = S_dec_block(skip1, f=seg_channels, name='out_unbound', batchnorm=False, relu=False)
    if seg_channels == 1:
        out = layers.Activation('sigmoid')(out_unb)
    else:
        out = layers.Activation('softmax')(out_unb)
    
    # testout = layers.Conv2D(1, 3, padding='same')(s_mri_in)
    return Model([s_mri_in], out, name='S')

def build_critic(mri_shape, seg_shape):
    def C_enc_block(_input, f, k=4, strides=2, batchnorm=True, clip=True, name='C_enc'):
        initializer = tk.initializers.RandomNormal(stddev=tf.math.sqrt(2./(k*k*f)))
        constraint = lambda x: tf.clip_by_value(x, clip_value_min=-0.05, clip_value_max=0.05) if clip else None
        out = layers.Conv2D(f, k, strides=(strides, strides), use_bias=not batchnorm, kernel_constraint=constraint, kernel_initializer=initializer, padding='same', name=name+'_conv')(_input) 
        if batchnorm:
            out = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name+'_bn')(out)
        out = layers.LeakyReLU(name=name+'_out')(out)
        return out
    
    c_mri_in = layers.Input(mri_shape)
    c_seg_in = layers.Input(seg_shape)
    
    mri_masked = layers.Concatenate()([c_mri_in, c_seg_in])
       
    enc1 = C_enc_block(mri_masked, f=64, name='C_enc_1', batchnorm=False)
    enc2 = C_enc_block(enc1, f=128, name='C_enc_2')
    enc3 = C_enc_block(enc2, f=256, name='C_enc_3')
    
    enc0fl = layers.Flatten()(mri_masked)
    enc1fl = layers.Flatten()(enc1)
    enc2fl = layers.Flatten()(enc2)
    enc3fl = layers.Flatten()(enc3)
    out = layers.Concatenate()([enc0fl, enc1fl, enc2fl, enc3fl])
    #out = tf.stack([enc1fl, tf.tile(enc3fl, [1,4])], axis=-1)
    return Model([c_mri_in, c_seg_in], out, name='C')
    
@tf.function
def loss_d(d_real, d_fake):
    return -tf.reduce_mean(tf.abs(d_real-d_fake))

@tf.function
def differentiable_dice_loss(g, p, axis=(1,2), eps=1e-6):
    return 1.0 - (eps + 2.0*tf.reduce_sum(p*g, axis=axis)) / (eps + tf.reduce_sum(tf.pow(p, 2), axis=axis) + tf.reduce_sum(tf.pow(g, 2), axis=axis))
    

@tf.function
def loss_g(d_real, d_fake, y_fake, y_true):
    return tf.reduce_mean(tf.abs(d_real-d_fake)) + tf.reduce_mean(differentiable_dice_loss(y_true, y_fake))
    