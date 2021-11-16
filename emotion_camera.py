import cv2
import numpy as np

import pickle

from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

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

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

all_emotions = {
        0: "Angry", 
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
}
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    _, frame = webcam.read()

    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(bw_img, 1.3, 5)

    for face in faces:
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 0), 2)
        face_img = cv2.resize(bw_img[y:(y+h), x:(x+w)], (48, 48))
        tens_face = valid_tsfm(face_img.reshape(48,48,1).astype(np.uint8)).unsqueeze(0).float()
        preds_val = model(tens_face)[0]
        pred_val = preds_val.argmax(0)
        pred_emotion = all_emotions[int(pred_val.numpy())]
        
        frame = cv2.putText(frame, pred_emotion, (x,y+h), font, 1, (100,255,0), 2)

    cv2.imshow('Webcam', frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
    # output the frame
    out.write(frame) 

    if cv2.waitKey(1) in [ord('q'), ord('x')] or \
        cv2.getWindowProperty('Webcam',cv2.WND_PROP_VISIBLE) < 1:
        break

webcam.release()
out.release()
cv2.destroyAllWindows()