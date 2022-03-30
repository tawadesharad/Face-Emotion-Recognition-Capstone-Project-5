# Live Class Monitoring System-Face Emotion Recognition 
------
## Introduction

Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context. To date, the most work has been conducted on automating the recognition of facial expressions from video, spoken expressions from audio, written expressions from text, and physiology as measured by wearables.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.

![1_Gkbf0PuGyxRht_ngfTC2cA](https://user-images.githubusercontent.com/85983864/160604997-121c90b7-9dc6-4574-84eb-deed303c34ca.gif)

## Problem Statements

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

## Dataset Information

The data comes from the past Kaggle competition “Challenges in Representation Learning: Facial Expression Recognition Challenge”. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
* This dataset contains 35887 grayscale 48x48 pixel face images.
* Each image corresponds to a facial expression in one of seven categories 

    Labels: 
    > * 0 - Angry :angry:</br>
    > * 1 - Disgust :anguished:<br/>
    > * 2 - Fear :fearful:<br/>
    > * 3 - Happy :smiley:<br/>
    > * 4 - Sad :disappointed:<br/>
    > * 5 - Surprise :open_mouth:<br/>
    > * 6 - Neutral :neutral_face:<br/>

Dataset link - https://www.kaggle.com/msambare/fer2013

## Dependencies

Python 3

Tensorflow 2.0

Streamlit

Streamlit-Webrtc

OpenCV

## Project Approch

#### Step 1. Build Model

We going to use three different models including pre-trained and Custom Models as follows:

 - Method1: Using DeepFace
 - Method2: Using ResNet
 - Method3: Custom CNN Model
#### Step 2. Real Time Prediction

And then we perform Real Time Prediction on our best model using webcam on Google colab itself.

  - Run webcam on Google Colab
  - Load our best Model
  - Real Time prediction
#### Step 3. Deployment

And lastly we will deploy it on three different plateform

   -  Streamlit Share
   -  Heroku
 
 ## Method 1: Using DeepFace

Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python.

The process of facial recognition starts with the human face and identifying its necessary facial features and patterns. A human face comprises a very basic set of features, such as eyes, nose, and mouth. Facial recognition technology learns what a face is and how it looks. This is done by using deep neural network & machine learning algorithms on a set of images with human faces looking at different angles or positions.
## Method 2: Transfer Learning - Prediction Using ResNet50
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. Reusing or transferring information from previously learned tasks to learning of new tasks has the potential to significantly improve the sample efficiency of a Data Scientist.

ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

![image](https://user-images.githubusercontent.com/85983864/160606438-17d89e68-7432-4cd5-8df3-875b35122096.png)

## Method 3: Custom CNN Model
A convolution network generally consists of alternate convolution and max-pooling operations. The output obtained after applying convolution operation is shrunk using max-pooling operation which is then used as an input for the next layer.

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics. The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.
![image](https://user-images.githubusercontent.com/85983864/160606721-a8d8c3b2-2338-4ab5-8aa4-ba8d7a3f2d48.png)


## Realtime Local Video Face Detection
The following photos and video contains the real time deploymnet of the app in local runtime 
![image](https://user-images.githubusercontent.com/85983864/160839340-050c4b9d-c81c-4d79-96fc-17eea561192d.png)
![image](https://user-images.githubusercontent.com/85983864/160839397-3b8487ab-6211-4743-85ed-4972fcd23836.png)
![image](https://user-images.githubusercontent.com/85983864/160839450-087342d6-49c8-4778-b19c-e4b9e785c484.png)
![image](https://user-images.githubusercontent.com/85983864/160839482-c09b30aa-3d42-41c6-af76-4fe4a7ae819e.png)
![image](https://user-images.githubusercontent.com/85983864/160839435-3eaad0c5-accc-4deb-859f-619dfe79824a.png)
![image](https://user-images.githubusercontent.com/85983864/160839538-990914f8-1219-4fdb-a4ba-a583feba45a5.png)
![image](https://user-images.githubusercontent.com/85983864/160839498-0237948a-06b7-4fcf-9b32-9a15ad7506fc.png)


## Deployment of Streamlit WebApp in Heroku and Streamlit

We have created front-end using Streamlit for webapp and used streamlit-webrtc which helped to deal with real-time video streams. Image captured from the webcam is sent to VideoTransformer function to detect the emotion. Then this model was deployed on heroku platform with the help of buildpack-apt which is necessary to deploy opencv model on heroku.


| Website | Link |
| ------ | ------ |
| Heroku | https://live-class-face-emotion.herokuapp.com/ |
| Streamlit | https://share.streamlit.io/tawadesharad/face-emotion-recognition-capstone-project-5/main/app.py |

## Conclusion
💪 trained the neural network and we achieved the highest validation accuracy of 66%.

😞 Pre Trained Model didn't gave appropriate result.

😊 The application is able to detect face location and predict the right expression while checking it on a local webcam.

🙂 The front-end of the model was made using streamlit for webapp and running well on local webapp link.

😎 Finally, we successfully deployed the Streamlit WebApp on Heroku and Streamlit share that runs on a web server.

😃 Our Model can succesfully detect face and predict emotion on live video feed as well as on an image.
