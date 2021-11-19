import streamlit as st
from aiortc.contrib.media import MediaPlayer
import av
import queue

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

import cv2
import numpy as np

import pickle

from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

st.set_page_config(page_title='Computer Vision, Explained', page_icon = "icon.ico")

stats = ([0.5],[0.5])
valid_tsfm = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(), 
    T.Normalize(*stats,inplace=True)
])

def conv_block(in_channels, out_channels, pool = False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace = True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def accuracy(output, labels):
    predictions, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

class expression_model(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet_expression(expression_model):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64) #went from 1 to 64 channels(400*64*48*48)
        self.conv2 = conv_block(64, 128, pool=True)  # batchsize*128*24*24
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))  #residual block one(no change in num_of_channel no pooling)
        
        self.conv3 = conv_block(128, 256, pool=True)  #400*256*12*12
        self.conv4 = conv_block(256, 512, pool=True)   #400*512*6*6
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))    #400*512*6*6
        
        self.conv5 = conv_block(512, 1024, pool=True)  #400*1025*3*3
        
        self.res3 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))    #400*1024*3*3
        
        self.classifier = nn.Sequential(nn.MaxPool2d(3), #400 * 1024 * 1 * 1
                                        nn.Flatten(), 
                                        nn.Linear(1024, num_classes)) 
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        
        out = self.classifier(out)
        return out


with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Emotion Detector")

st.sidebar.title("[Cason Wight](casonwight.com)")

st.sidebar.write("""
The model used in this app is trained from Google image searches of the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Due to this training methodology, you might need to make cartoonishly emotive faces like what a Google search might give.

For more details on how computer vision works generally, see my [blog post](https://cv-tutorial.herokuapp.com/).
""")

def app_emotion_detection():
    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        def __init__(self):
            
            self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            self.all_emotions = {
                    0: "Angry", 
                    1: "Disgust",
                    2: "Fear",
                    3: "Happy",
                    4: "Sad",
                    5: "Surprise",
                    6: "Neutral"
            }
            self.font = cv2.FONT_HERSHEY_SIMPLEX

        def _annotate_image(self, frame):
            bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(bw_img, 1.3, 5)

            for face in faces:
                x, y, w, h = face
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 0), 2)
                face_img = cv2.resize(bw_img[y:(y+h), x:(x+w)], (48, 48))
                tens_face = valid_tsfm(face_img.reshape(48,48,1).astype(np.uint8)).unsqueeze(0).float()
                preds_val = model(tens_face)[0]
                pred_val = preds_val.argmax(0)
                pred_emotion = self.all_emotions[int(pred_val.numpy())]

                frame = cv2.putText(frame, pred_emotion, (x,y+h), self.font, 1, (100,255,0), 2)
            return frame

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            annotated_image = self._annotate_image(cv2.flip(image, 1))

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

  
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


app_emotion_detection()

st.write("""
This app employs a haar cascade model (native to python's computer-vision library, `cv2`) and a convolutional neural net (CNN). 
The cascade model detects faces in a frame. The CNN takes these faces and assigns the most likely emotion.

Each frame is processed as a $w\\times h\\times 3$ tensor. 
The face detection model gives $n$ smaller black and white images (one for each of the $n$ faces that are detected), which are resized.
When faces are detected, the frame is also annotated with a box surrounding the likely face.

Each of the faces is run through the CNN to get probabilities for all possible emotions. 
The emotion with the highest probability is kept and shown on the original frame..

""")



st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<p style="text-align: center;">
<a href="mailto:cason.wight@gmail.com?subject = Feedback&body = Message" target="_blank" class="fa fa-send fa-lg fa-fw"></a>
<a href="https://www.linkedin.com/in/casonwight/" target="_blank" class="fa fa-linkedin fa-lg fa-fw"></a>
<a href="https://github.com/casonwight" target="_blank" class="fa fa-github fa-lg fa-fw"></a>
</p>

<a href="https://casonwight.com/Cason%20Wight%20Resume.pdf" download target="_blank"><p style="text-align:center">See my Resume</p></a>
<p style="text-align:center; font-size:75%; color:grey">Made by Cason Wight</p>

    
    """, unsafe_allow_html=True)