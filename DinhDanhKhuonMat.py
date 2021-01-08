#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Documentation: https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/

import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir

import face_recognition
import os
import glob
from numpy import random


# In[2]:


#Import hinh mau 

faces_encodings = []

faces_names = []

cur_direc = os.getcwd() #lay dia chi hien tai

path = os.path.join(cur_direc, 'data\\faces\\')

list_of_files = [f for f in glob.glob(path+'*.jpg')]

number_files = len(list_of_files)

names = list_of_files.copy()

#Import Icon Gioi Tinh
enableGenderIcons = True

male_icon = cv2.imread("data/male.jpg")
male_icon = cv2.resize(male_icon, (40, 40))

female_icon = cv2.imread("data/female.jpg")
female_icon = cv2.resize(female_icon, (40, 40))


# In[3]:


for i in range(number_files):
    names[i] = names[i][42:]
    names[i] = names[i].replace(' ','')
    names[i] = names[i].replace('.jpg','')
    names[i] = names[i].replace('.JPG','')
    names[i] = ''.join([i for i in names[i] if not i.isdigit()])
    
    try:
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    except:
        continue
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Tao mang luu ten nguoi va xu ly ten file
    
    faces_names.append(names[i])

#Tao cac bien
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print('DONE')


# In[4]:


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model

def ageModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)
    
    #Load dataset tuổi tác
    #you can find the pre-trained weights for age prediction here: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
    age_model.load_weights("models/age_model_weights.h5")

    return age_model

def genderModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)
    
    #Load dataset giới tính
    #you can find the pre-trained weights for gender prediction here: https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
    gender_model.load_weights("models/gender_model_weights.h5")

    return gender_model

age_model = ageModel()
gender_model = genderModel()

#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])


# In[ ]:


cap = cv2.VideoCapture(0) #Capture Video từ Webcam
while(True):
    ret, img = cap.read()
    #Thu nho hinh anh xuong 1/4 de xu ly nhanh hon
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    #Doi mau BGR sang RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    
    if process_this_frame:
        face_locations = face_recognition.face_locations( rgb_small_frame)
        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
        face_names = []
        
        #So sanh guong mat voi hinh anh trong thu vien
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces (faces_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]          
            face_names.append(name)

    process_this_frame = not process_this_frame

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h), name in zip(faces, face_names):
        
        if w > 130: #Bỏ qua những khuôn mặt chiếm ít khung hình

            
            cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1) #Vẽ hình chữ nhật quanh đói tượng

            #Chiếc xuất khuôn mặt định dạng được
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #cắt khuôn mặt 

            try:
                #Dataset của độ tuổi và giới tính có biên độ 40% quanh khuôn mặt vì vậy phải mở rộng khuôn mặt định dạng được.
                margin = 30
                margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
                detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
            except:
                print("detected face has no margin")

            try:
                #vgg-face expects inputs (224, 224, 3)
                detected_face = cv2.resize(detected_face, (224, 224))

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                #Du doan tuoi
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))

                gender_distribution = gender_model.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)

                if gender_index == 0: gender = "F"
                else: gender = "M"

                #Định dạng khung nền
                info_box_color = (186, 191, 240)
                scaling = len(name)*20 + 50

                #triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
                triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
                cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+ scaling,y-90),info_box_color,cv2.FILLED)

                #Định dạng Giới Tính, Tuổi Tác và Danh Xưng
                cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(img, name, (x+int(w/2)+45, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if enableGenderIcons:
                    if gender == 'M': gender_icon = male_icon
                    else: gender_icon = female_icon

                    img[y-75:y-75+male_icon.shape[0], x+int(w/2)-45:x+int(w/2)-45+male_icon.shape[1]] = gender_icon
                else:
                    cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                print("exception",str(e))

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #Ấn phím Q để thoát
        break

#Tắt OpenCV
cap.release()
cv2.destroyAllWindows()


# In[ ]:




