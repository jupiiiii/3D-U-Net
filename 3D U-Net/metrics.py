!pip install volumentations-3D
!pip install MedPy
import numpy as np
from volumentations import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import keras.backend as K
from scipy import ndimage
from medpy.metric.binary import hd95


def dice_coef(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    """
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_wt(y_true, y_pred):
    y_true_wt = y_true[:,:,:,:,1] + y_true[:,:,:,:,2] + y_true[:,:,:,:,3]
    y_pred_wt = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,2] + y_pred[:,:,:,:,3]
    return dice_coef(y_true_wt, y_pred_wt)

def dice_coef_tc(y_true, y_pred):
    y_true_tc = y_true[:,:,:,:,1] + y_true[:,:,:,:,3]
    y_pred_tc = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,3]
    return dice_coef(y_true_tc, y_pred_tc)

def dice_coef_et(y_true, y_pred):
    y_true_et = y_true[:,:,:,:,3]
    y_pred_et = y_pred[:,:,:,:,3]
    return dice_coef(y_true_et, y_pred_et)

#def precision(y_true, y_pred):
    #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #precision = true_positives / (predicted_positives + K.epsilon())
   # return precision

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (actual_positives + K.epsilon())
    return sensitivity

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / negatives

#def precision_wt(y_true, y_pred):
 # y_true_wt = y_true[:,:,:,:,1] + y_true[:,:,:,:,2] + y_true[:,:,:,:,3]
 # y_pred_wt = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,2] + y_pred[:,:,:,:,3]
 # return precision(y_true_wt, y_pred_wt)

#def precision_tc(y_true, y_pred):
 # y_true_tc = y_true[:,:,:,:,1] + y_true[:,:,:,:,3]
 # y_pred_tc = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,3]
 # return precision(y_true_tc, y_pred_tc)

#def precision_et(y_true, y_pred):
 # y_true_et = y_true[:,:,:,:,3]
 # y_pred_et = y_pred[:,:,:,:,3]
 # return precision(y_true_et, y_pred_et)

def sensitivity_wt(y_true, y_pred):
  y_true_wt = y_true[:,:,:,:,1] + y_true[:,:,:,:,2] + y_true[:,:,:,:,3]
  y_pred_wt = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,2] + y_pred[:,:,:,:,3]
  return sensitivity(y_true_wt, y_pred_wt)

def sensitivity_tc(y_true, y_pred):
  y_true_tc = y_true[:,:,:,:,1] + y_true[:,:,:,:,3]
  y_pred_tc = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,3]
  return sensitivity(y_true_tc, y_pred_tc)

def sensitivity_et(y_true, y_pred):
  y_true_et = y_true[:,:,:,:,3]
  y_pred_et = y_pred[:,:,:,:,3]
  return sensitivity(y_true_et, y_pred_et)

def specificity_wt(y_true, y_pred):
  y_true_wt = y_true[:,:,:,:,1] + y_true[:,:,:,:,2] + y_true[:,:,:,:,3]
  y_pred_wt = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,2] + y_pred[:,:,:,:,3]
  return specificity(y_true_wt, y_pred_wt)

def specificity_tc(y_true, y_pred):
  y_true_tc = y_true[:,:,:,:,1] + y_true[:,:,:,:,3]
  y_pred_tc = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,3]
  return specificity(y_true_tc, y_pred_tc)

def specificity_et(y_true, y_pred):
  y_true_et = y_true[:,:,:,:,3]
  y_pred_et = y_pred[:,:,:,:,3]
  return specificity(y_true_et, y_pred_et)

# Set voxel spacing according to your dataset
# Our Dataset voxel spacing is 0.1 mm but we 
# set voxel spacing to default which is unit
# which means we have to devide our HD95 results by 10

def HD95(y_true, y_pred):
  #y_true = tf.keras.backend.eval(y_true)
  #y_pred = tf.keras.backend.eval(y_pred)
  return hd95(y_true, y_pred)

def HD95_wt(y_true, y_pred):
  #y_true = tf.keras.backend.eval(y_true)
  #y_pred = tf.keras.backend.eval(y_pred)
  y_true_wt = y_true[:,:,:,:,1] + y_true[:,:,:,:,2] + y_true[:,:,:,:,3]
  y_pred_wt = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,2] + y_pred[:,:,:,:,3]
  return hd95(y_true_wt, y_pred_wt)

def HD95_tc(y_true, y_pred):
  #y_true = tf.keras.backend.eval(y_true)
  #y_pred = tf.keras.backend.eval(y_pred)
  y_true_tc = y_true[:,:,:,:,1] + y_true[:,:,:,:,3]
  y_pred_tc = y_pred[:,:,:,:,1] + y_pred[:,:,:,:,3]
  return hd95(y_true_tc, y_pred_tc)

def HD95_et(y_true, y_pred):
  #y_true = tf.keras.backend.eval(y_true)
  #y_pred = tf.keras.backend.eval(y_pred)
  #y_true_et_binary = np.where(y_true > 0.5, 1, 0)
  #y_pred_et_binary = np.where(y_pred > 0.5, 1, 0)

  try:
    y_true_et = y_true[:,:,:,:,3] #+ y_true[:,:,:,:,0]
    y_pred_et = y_pred[:,:,:,:,3] #+ y_pred[:,:,:,:,0]
  
    return hd95(y_true_et, y_pred_et)
  except RuntimeError:
    return 1

#def HD95_zero(y_true, y_pred):
  #y_true = tf.keras.backend.eval(y_true)
  #y_pred = tf.keras.backend.eval(y_pred)
  #y_true_et_binary = np.where(y_true > 0.5, 1, 0)
  #y_pred_et_binary = np.where(y_pred > 0.5, 1, 0)

  #y_true_zero = y_true[:,:,:,:,0]
  #y_pred_zero = y_pred[:,:,:,:,0]
  
  #return hd95(y_true_zero, y_pred_zero)
def get_augmentation(patch_size):
    return Compose([
        Rotate((0, 90, 180, 270), (0, 0, 0), (0, 0, 0), p= 1),
        #ElasticTransform((0, 1), interpolation=2, p=0.2),
        Flip(0, p=1/3),
        Flip(1, p=1/3),
        Flip(2, p=1/3)
        #GaussianNoise(var_limit=(0, 0.001), p=0.2)
        #Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0)
    ], p=1.0)
