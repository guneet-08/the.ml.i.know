# -*- coding: utf-8 -*-

# DIABETIC RETINOPATHY DETECTION USING TRANSFER LEARNING 
# MODELS USED : VGG16, ResNet50, Inception V3   (Pre-Developed Image processing libraries available via Keras)


"""
Created on Mon Jun  6 00:22:39 2022

@author: Guneet
"""

"""
OVERVIEW : In heathcare, diabetic retinopathy is a manual process that 
involves a trained physician examining the color fundus retina images.
This project will help to automate this process and classify the severity 
of the condition of the retina, with respect to diabetic retinipathy.
The conditions into which classification will be done : 
0 : No diabetic retinopathy
1 : Mild diabetic retinopathy
2 : Moderate diabetic retinopathy
3 : Severe diabetic retinopathy
4 : Proliferative diabetic retinopathy

DATASET LINK : "https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=test.zip.001" 

"""

# importing the required libraries 

# SYSTEM LIBRARIES
import numpy as np
np.random.seed(1000)
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


# ML MODEL MAKING LIBRARIES
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss 
import keras
from keras import __version__ as keras_version
from keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.ResNet50
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.applications.resnet50 import preprocess_input
import h5py
import argparse
from sklearn.externals import joblib
import json


# defining a class for working through the entire process of Transfer Learning

class TransferLearning:
    
#     initialization

    def __init__(self):
        
#         parsing arguments and help messages for the inputs
        parser = argparse.ArgumentParser(description='Process the inputs')
        parser.add_argument('--path',help='image directory')
        parser.add_argument('--class_folders',help='class images folder names')
        parser.add_argument('--dim',type=int,help='Image dimensions to process')
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-4)
        parser.add_argument('--batch_size',type=int,help='batch size')
        parser.add_argument('--epochs',type=int,help='no of epochs to train')
        parser.add_argument('--initial_layers_to_freeze',type=int,help='the initial layers to freeze')
        parser.add_argument('--model',help='Standard Model to load',default='InceptionV3')
        parser.add_argument('--folds',type=int,help='num of cross validation folds',default=5)
        parser.add_argument('--outdir',help='output directory')
        
#         storing the parsed values into args
        args = parser.parse_args()
        
#         defining the variables to be used within the class
        self.path = args.path
        self.class_folders = json.loads(args.class_folders)
        self.lr = float(args.lr)
        self.batch_size = int(args.batch_size)
        self.epochs = int(args.epochs)
        self.initial_layers_to_freeze = int(args.initial_layers_to_freeze)
        self.model = args.model
        self.folds = int(args.folds)
        self.outdir = args.outdir
        
#     Getting the Path of the Image from the Dataset

    def get_im_cv2(self,path,dim=224):
        img = cv2.imread(path)
        resized = cv2.resize(img,(dim,dim), cv2.INTER_LINEAR)
        return resized
    
#     Pre Processing the images based on the ImageNet pre-trained model 
#     (Image Transformation)

    def pre_process(self,img):
        img[:,:,0] = img[:,:,0] = 103.939
        img[:,:,1] = img[:,:,0] - 116.779
        img[:,:,2] = img[:,:,0] - 123.68
        return img
    
#     Function to build x, y in numpy (Format based on the Train/Validation 
#     Datasets)

    def read_data(self,class_folders,path,num_class,dim,train_val = 'train'):
        print(train_val)
        train_X,train_y = [],[] 
        for c in class_folders:
            path_class = path + str(train_val) + '/' + str(c)
            file_list = os.listdir(path_class) 
            for f in file_list:
                img = self.get_im_cv2(path_class + '/' + f)
                img = self.pre_process(img)
                train_X.append(img)
                label = int(c.split('class')[1])
                train_y.append(int(label))
        train_y = keras.utils.np_utils.to_categorical(np.array(train_y),num_class) 
        return np.array(train_X),train_y
    
#     Creating the Function to process the images based on the InceptionV3 model of image processing (default weights being used)   

    def inception_pseudo(self,dim=224,freeze_layers=30,full_freeze='N'):
        model = InceptionV3(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final
    
#     ResNet50 Model for transfer Learning (default weights being used)

    def resnet_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = ResNet50(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final

#     VGG16 Model for transfer Learning (default weights being used, again)

    def VGG16_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = VGG16(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final
    
    
#     Training the final model based on which of these models is selected (BY default : ResNet50 is used) 
#     Process used for training puposes is K-Fold Cross Validation
    
    def train_model(self,train_X,train_y,n_fold=5,batch_size=16,epochs=40,dim=224,lr=1e-5,model='ResNet50'):
        model_save_dest = {}
        k = 0
        kf = KFold(n_splits=n_fold, random_state=0, shuffle=True)

        for train_index, test_index in kf.split(train_X):


            k += 1 
            X_train,X_test = train_X[train_index],train_X[test_index]
            y_train, y_test = train_y[train_index],train_y[test_index]
            
            if model == 'Resnet50':
                model_final = self.resnet_pseudo(dim=224,freeze_layers=10,full_freeze='N')
            
            if model == 'VGG16':
                model_final = self.VGG16_pseudo(dim=224,freeze_layers=10,full_freeze='N') 
            
            if model == 'InceptionV3':
                model_final = self.inception_pseudo(dim=224,freeze_layers=10,full_freeze='N')
            
#           Preprocessing these images using the inbuilt ImageDataGeneratior (in Keras) and setting the Hyper Parameters 
            datagen = ImageDataGenerator(
                    horizontal_flip = True,
                    vertical_flip = True,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    channel_shift_range=0,
                    zoom_range = 0.2,
                    rotation_range = 20)
              
#           Setting up the parameters for model validation and training    
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model_final.compile(optimizer=adam, loss=["categorical_crossentropy"],metrics=['accuracy'])
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,
                                                           patience=3, min_lr=0.000001)
            
#           In case of callbacks, moving through the probable raisable errors and bypassing them
            callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1),
                        CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False),reduce_lr,
                        ModelCheckpoint(
                                'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
                                monitor='val_loss', mode='min',
                                save_best_only=True,
                                verbose=1)]
            
#           Fitting the final model with the above parameters 
            model_final.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
            steps_per_epoch=X_train.shape[0]/batch_size,epochs=epochs,verbose=1,
            validation_data=(X_test,y_test),callbacks=callbacks,
                                          class_weight={0:0.012,1:0.12,2:0.058,3:0.36,4:0.43})
    
#           saving the model test details in to a file for further optimization     
            model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'
            del model_final
            f = h5py.File(model_name, 'r+')
            del f['optimizer_weights']
            f.close()
            model_final = keras.models.load_model(model_name)
            model_name1 = self.outdir + str(model) + '___' + str(k) 
            model_final.save(model_name1)
            model_save_dest[k] = model_name1
                
        return model_save_dest
    
#     Hold out dataset validation function

    def inference_validation(self,test_X,test_y,model_save_dest,n_class=5,folds=5):
        pred = np.zeros((len(test_X),n_class))

        for k in range(1,folds + 1):
            model = keras.models.load_model(model_save_dest[k])
            pred = pred + model.predict(test_X)
        
        pred = pred/(1.0*folds) 
        pred_class = np.argmax(pred,axis=1) 
        act_class = np.argmax(test_y,axis=1)
        accuracy = np.sum([pred_class == act_class])*1.0/len(test_X)
        kappa = cohen_kappa_score(pred_class,act_class,weights='quadratic')
        return pred_class,accuracy,kappa   
   

#   THE MAIN FUNCTION
    def main(self):
        start_time = time.time()
        print('Data Processing..')
        self.num_class = len(self.class_folders)
        train_X,train_y = self.read_data(self.class_folders,self.path,self.num_class,self.dim,train_val='train')
        self.model_save_dest = self.train_model(train_X,train_y,n_fold=self.folds,batch_size=self.batch_size,
                                                        epochs=self.epochs,dim=self.dim,lr=self.lr,model=self.model)
        print("Model saved to dest:",self.model_save_dest)
        test_X,test_y = self.read_data(self.class_folders,self.path,self.num_class,self.dim,train_val='validation')
        _,accuracy,kappa = self.inference_validation(test_X,test_y,self.model_save_dest,n_class=self.num_class,folds=self.folds)
        joblib.dump(self.model_save_dest,self.outdir  + "dict_model.pkl")
        
#       The metrics for each type of model to compare the accurary for each i.e. ResNet50, VGG16 and InceptionV3
        print("-----------------------------------------------------")
        print("Kappa score:", kappa)
        print("accuracy:", accuracy) 
        print("End of training")
        print("-----------------------------------------------------")
        print("Processing Time",time.time() - start_time,' secs')
    
#   Running the entire procees on an object of TransferLerning Class    
if __name__ == "__main__":
    obj = TransferLearning()
    obj.main()

# END OF EXECUTABLE CODE

# NOTE that the downloaded DataSet has to be stored in a folder as created above in the __init__ method before executing this code.
        
        
