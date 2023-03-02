import keras
from keras.models import Model
from keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda

class unet_3d(object):

  def __init__(self, input_shape, n_filters=8, dropout=0.2, batch_norm=True):
    
    self.input_shape = input_shape
    self.n_filters = n_filters
    self.dropout = dropout
    self.batch_norm = batch_norm
    self.model = self.compile_unet_3d()

  def compile_unet_3d(self):

    """
    Compile the 3d Model
    """
    inputs = Input(self.input_shape)
    out = self.unet(inputs)
    model = Model(inputs=inputs, outputs=out)

    model.compile(run_eagerly=True,loss = dice_coef_loss, optimizer = keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ['accuracy',tf.keras.metrics.OneHotMeanIoU(num_classes=4), dice_coef,
                         dice_coef_wt, dice_coef_tc ,dice_coef_et,
                         sensitivity, specificity, 
                         sensitivity_wt,
                         sensitivity_tc, sensitivity_et,
                         specificity_wt, specificity_tc,
                         specificity_et, HD95, HD95_wt,
                         HD95_tc, HD95_et
                         ])
    print("The 3D Model is compiled!!!")
    return model
  
  def unet(self,input_img, n_filters = 8, dropout = 0.2, batch_norm = True):

    c1 = self.conv_block(input_img,n_filters,3,batch_norm)
    p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
    p1 = Dropout(dropout)(p1)

    c2 = self.conv_block(p1,n_filters*2,3,batch_norm);
    p2 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c2)
    p2 = Dropout(dropout)(p2)

    c3 = self.conv_block(p2,n_filters*4,3,batch_norm);
    p3 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c3)
    p3 = Dropout(dropout)(p3)
  
    c4 = self.conv_block(p3,n_filters*8,3,batch_norm);
    p4 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c4)
    p4 = Dropout(dropout)(p4)
  
    c5 = self.conv_block(p4,n_filters*16,3,batch_norm);

    u6 = Conv3DTranspose(n_filters*8, (3,3,3), strides=(2, 2, 2), padding='same')(c5);
    u6 = concatenate([u6,c4]);
    c6 = self.conv_block(u6,n_filters*8,3,batch_norm)
    c6 = Dropout(dropout)(c6)
    u7 = Conv3DTranspose(n_filters*4,(3,3,3),strides = (2,2,2) , padding= 'same')(c6);

    u7 = concatenate([u7,c3]);
    c7 = self.conv_block(u7,n_filters*4,3,batch_norm)
    c7 = Dropout(dropout)(c7)
    u8 = Conv3DTranspose(n_filters*2,(3,3,3),strides = (2,2,2) , padding='same')(c7);
    u8 = concatenate([u8,c2]);

    c8 = self.conv_block(u8,n_filters*2,3,batch_norm)
    c8 = Dropout(dropout)(c8)
    u9 = Conv3DTranspose(n_filters,(3,3,3),strides = (2,2,2) , padding='same')(c8);

    u9 = concatenate([u9,c1]);

    c9 = self.conv_block(u9,n_filters,3,batch_norm)
    out = Conv3D(4, (1, 1,1), activation='softmax')(c9)

    return out
  
  def conv_block(self, input_mat, num_filters, kernel_size, batch_norm):
    X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
    if batch_norm:
      X = BatchNormalization()(X)
  
    X = Activation('relu')(X)

    X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
    if batch_norm:
      X = BatchNormalization()(X)
  
    X = Activation('relu')(X)
  
    return X
    
