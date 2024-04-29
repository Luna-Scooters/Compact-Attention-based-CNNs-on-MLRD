
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import pandas as pd
import cv2
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D, Input, DepthwiseConv2D, ReLU, AvgPool2D,GlobalAveragePooling2D, Concatenate, Add, GlobalAvgPool2D, Reshape, Permute, Lambda, GlobalMaxPool2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import tempfile
# import tf.nn.relu6 as ReLU6
from sklearn.preprocessing import MultiLabelBinarizer
# from keras_flops import get_flops
tf.keras.backend.set_image_data_format('channels_last')
import wandb
from wandb.keras import WandbCallback



# -------------------------------------------------------fetching all label file paths----------------------------------------------------
filepath='MLRD/labels/'
def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)        

paths = list_files(filepath, '.csv')
print(len(paths))
paths = paths[:33]+paths[34:]   #Adjust the number of labels to be used
# -------------------------------------------------------declaring test dataset list ----------------------------------------------------

test_videos=["Dublin01/journeys/1F27CA09DAB5FF65/2022-06-10/journey_021.csv", 
  
     
                     "Dublin01/journeys/1F27CA09DAB5FF65/2022-06-17/journey_008.csv", 
                     "Dublin01/journeys/1F27CA09DAB5FF65/2022-06-23/journey_003.csv", 
                     "Dublin01/journeys/1F27CA09DAB5FF65/2022-06-23/journey_005.csv", 
                     "Dublin01/journeys/041BBA09DAB5FF65/2022-05-24/journey_003.csv", 
                     "Northampton02/Northampton02-vid-14-15-17.csv",
                     "Northampton04/VOI_2022-02-02_Test-1.csv", 
                     "Northampton04/VOI_2022-02-02_Test-3.csv",
                     "Northampton04/VOI_2022-02-02_Test-4.csv",
                     "Stockholm/GH010129.csv",
                     "Paris/Paris_Dott02.csv",
                     "Paris/Paris_Dott03.csv",
                    ]
# -------------------------------------------------------setting up test and train datasets ----------------------------------------------------
relative_path = 'MLRD/labels/'
test_paths = [relative_path + journey for journey in test_videos]
train_paths = [path for path in paths if path not in test_paths]
# -------------------------------------------------------data preparation before training----------------------------------------------------
def df_preprocessing_function(paths):
    main_path = 'MLRD/'
    dataframes_list = []
    
    #This function adds zeros before each frame name to match it with our classification dataset
    def add_zero(x):
        if len(str(x))==1:
            return '00000'+str(x)
        elif len(str(x))==2:
            return '0000'+str(x)
        elif len(str(x))==3:
            return '000'+str(x)

    def df_preprocessing(csv_path):
        df = pd.read_csv(csv_path)
        df = df.drop(['on', 'off', 'Unnamed: 17', 'asphalt', 'concrete', 'cobblestone', 'dirt', 'grass', 'day', 'night', 'indoor', 'sunny', 'cloudy', 'raining', 'sidewalk'], axis=1)
        df['frame']=df['frame'].apply(add_zero) 
        image_filepath= main_path + 'less_frames/' + csv_path[68:-4] #Adjust the path according to your directory structure
        #adding the img_name column containing absolute path of the image for ImageDataGenerator
        df['img_name'] = image_filepath + '/' + df['frame'] + ".png"
        return df

    df={}

    for i in range(len(paths)):
        df["df{0}".format(i)] = df_preprocessing(paths[i])
        dataframes_list.append(df["df{0}".format(i)])
    return df

# -------------------------------------------------------train and test dataframes preparation----------------------------------------------------

'''
train_df_paths will have all the training dataframes and test_df_paths will have all the test dataframes
'''

train_df_paths = df_preprocessing_function(train_paths)
test_df_paths = df_preprocessing_function(test_paths)

'''
merging all the separate dataframes into a single dataframe by concatenating
'''

train_df_merged = pd.concat(train_df_paths.values(), ignore_index=True)
test_df_merged = pd.concat(test_df_paths.values(), ignore_index=True)

train_df_merged_copy = train_df_merged
test_df_merged_copy = test_df_merged


# '''
# few labels and frames are missing so dropping those records
# '''


train_df_merged_copy = train_df_merged_copy.dropna(how='any')
test_df_merged_copy = test_df_merged_copy.dropna(how='any')


# print(train_df_merged_copy.head())
# print(train_df_merged_copy.columns)
    
print("Number of frames with the ROAD label in TRAIN DS: ",train_df_merged_copy[train_df_merged_copy.road == 1].shape[0])
print("Number of frames with the BIKELANE label IN TRAIN DS:",train_df_merged_copy[train_df_merged_copy.bikelane == 1].shape[0])


print("Number of frames with the ROAD label in TEST DS: ",test_df_merged_copy[test_df_merged_copy.road == 1].shape[0])
print("Number of frames with the BIKELANE label in TEST DS: ",test_df_merged_copy[test_df_merged_copy.bikelane == 1].shape[0])

print(train_df_merged_copy['img_name'][0])
# -------------------------------------------------------data preparation and augmentation----------------------------------------------------

train_df, val_df = train_test_split(train_df_merged_copy, test_size=0.25, random_state=42)

class_names = ["road", "bikelane"]

num_classes = len(class_names)

train_img_data_generator = ImageDataGenerator(rescale=1./255, 
                                        horizontal_flip = True,
                                        brightness_range = (0.2, 0.5),
                                       )


test_img_data_generator = ImageDataGenerator(rescale=1./255)



## Recreate datasets from dataframe
train_generator = train_img_data_generator.flow_from_dataframe(dataframe=train_df,
                                                    directory=None,
                                                    x_col="img_name",
                                                    y_col= class_names,
                                                    target_size=(224, 224),
                                                    class_mode='raw',
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    shuffle=True,                                    
                                                    )


val_generator = train_img_data_generator.flow_from_dataframe(dataframe=val_df,
                                                    directory=None,
                                                    x_col="img_name",
                                                    y_col=class_names,
                                                    target_size=(224, 224),
                                                    class_mode='raw',
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    shuffle=True,
                                                    )

test_generator = test_img_data_generator.flow_from_dataframe(dataframe=test_df_merged_copy,
                                                    directory=None,
                                                    x_col="img_name",
                                                    y_col= class_names,
                                                    target_size=(224, 224),
                                                    class_mode='raw',
                                                    color_mode='rgb',
                                                    batch_size=1,
                                                    shuffle=False,
                                                    )




# ____________________________________________________________________________________________________________________________________________






def mobilenet_0_1():      #Modified MobileNetV2 architecture code is shared separately. Please replace /keras_applications/mobilenet_v2.py file with that. 
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=0.1,
    include_top=False,
    weights=None,
    )

    GAP = GlobalAveragePooling2D(name='final_gap_layer')(
        model.layers[-1].output)
    FC = Dense(num_classes, activation='sigmoid', name='output_layer')(GAP) 
    
    final_model = Model(inputs=model.inputs, outputs=FC)

    return final_model

fp32_model = mobilenet_0_1()

tf.keras.utils.plot_model(
    fp32_model,
    to_file='wandb-MNV2-CA-cropped-0.1.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
    show_trainable=False
)


print("LETS SEE NOW: ",fp32_model.summary())
# flops = get_flops(fp32_model, batch_size=1)
# print(f"FLOPS: {flops / 10 ** 9:.05} G")
# print(f"FLOPS: {flops / 10 ** 6:.05} M")


f1 = tfa.metrics.F1Score(num_classes=num_classes, average='weighted', threshold=0.5)

# MLCM = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes)

auc = tf.keras.metrics.AUC(
                name="auc",
                multi_label=True
            )

#https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
#It is the harmonic mean of precision and recall. Output range is [0, 1]. Works for both multi-class and multi-label classification.

# MLCM = tfa.metrics.MultiLabelConfusionMatrix(num_classes=2)
'''
1. average=micro:

True positivies, false positives and
    false negatives are computed globally.
    
2. average=macro:

True positivies, false positives and
    false negatives are computed for each class
    and their unweighted mean is returned.
'''


from tensorflow.keras.callbacks import Callback

class PlotHistory(Callback):
    def __init__(self, folder_path):
        super(PlotHistory, self).__init__()
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        history = self.model.history.history
        
        # Update and save the latest plots for each metric
        self.plot_and_save(history, 'accuracy', 'val_accuracy', 'Model Accuracy', 'Accuracy')
        self.plot_and_save(history, 'f1_score', 'val_f1_score', 'Model F1 Score', 'F1 Score')
        self.plot_and_save(history, 'auc', 'val_auc', 'Model AUC', 'AUC')
        self.plot_and_save(history, 'loss', 'val_loss', 'Model Loss', 'Loss')

        # Save the current epoch number in a text file
        with open(os.path.join(self.folder_path, 'latest_epoch.txt'), 'w') as file:
            file.write(f'Latest Epoch: {epoch}')

    def plot_and_save(self, history, train_key, val_key, title, ylabel):
        if train_key in history and val_key in history:
            plt.plot(history[train_key])
            plt.plot(history[val_key])
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(self.folder_path, f'{ylabel.lower().replace(" ", "_")}.png'))
            plt.clf()

# Specify the folder path
folder_path = 'Models/wandb-MNV2-CA-cropped-224-224'

# Instantiate the custom callback with the folder path
plot_history_callback = PlotHistory(folder_path)



# ==================================++++++++++++WEIGHTS AND BIASES+++++++++++====================================
wandb.login()
wandb.init(project="attentions-in-vision", config={
    "learning_rate": 0.001,     #Replace with your hyperparameters here
    "momentum": 0.9,
    "epochs": 80,
    "batch_size": train_generator.batch_size,
    "loss_function": "BinaryFocalCrossentropy",
    "optimizer": "Adam",
    "experiment_id": wandb.util.generate_id()
})
# ==================================++++++++++++WEIGHTS AND BIASES+++++++++++====================================




# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=np.sqrt(0.1), patience=30)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr = 1e-6, mode='min', verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('Models/wandb-MNV2-CA-cropped-224-224/wandb-MNV2-CA-cropped.h5', 
                                                monitor='val_loss', 
                                                mode='min', 
                                                verbose=1, 
                                                save_best_only=True)

optimizer = Adam(learning_rate=0.001)

fp32_model.compile(optimizer=optimizer, 
                        loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0), 
                        metrics=[['accuracy'],f1, auc])

history=fp32_model.fit(train_generator, 
                              steps_per_epoch=len(train_generator),
                              epochs=80, 
                              validation_data=val_generator,
                              validation_steps=len(val_generator),
                              verbose=1,
                              callbacks=[checkpoint, reduce_lr, WandbCallback(), plot_history_callback])

wandb.finish()
