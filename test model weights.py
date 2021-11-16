from torch import nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import pickle
import cv2
import numpy as np
from skimage.exposure import rescale_intensity


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

with open("emotion_detector//emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

face = cv2.resize(cv2.imread("emotion_detector//surprise_face.png", -1), (48, 48))
#face_frmt = valid_tsfm(face.reshape(48,48,1).astype(np.uint8)).unsqueeze(0).float()

kernels = model.conv1[0].weight.data.clone().numpy()[np.array([28, 24, 26]),0,:,:]
kernels*= 1.5

def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2

    images = [cv2.copyMakeBorder(image[:,:,channel], pad, pad, pad, pad,
        cv2.BORDER_REPLICATE) for channel in range(image.shape[2])]
    outputs = [np.zeros((iH, iW), dtype="float32") for channel in range(image.shape[2])]
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            rois = [image[y - pad:y + pad + 1, x - pad:x + pad + 1] for image in images]
            ks = [(roi * kernel).sum() for roi in rois]
            for index in range(len(outputs)):
                outputs[index][y - pad, x - pad] = ks[index]

    # rescale the output image to be in the range [0, 255]
    outputs = [rescale_intensity(output, in_range=(0, 255)) for output in outputs]
    outputs = [(output * 255).astype("uint8") for output in outputs]
    # return the output image
    output = np.stack(outputs, axis = 2)
    return output

bgr_face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

for i, kernel in enumerate(kernels):
    #cv2.imshow("conv" + str(i), convolve(bgr_face, kernel))
    cv2.imwrite("conv" + str(i) + ".png", convolve(bgr_face, kernel))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

surprise_face = cv2.imread("emotion_detector//surprise_face.png", -1)
surprise_face = cv2.resize(surprise_face, (144, 144))

background = np.zeros(surprise_face.shape, dtype = "uint8") + 255
background = cv2.resize(background, (0,0), fx = 5, fy = 1.5)

background[36:(36 + 144), :144] = surprise_face

background = cv2.arrowedLine(background, (160,108), (185,108), (0,0,0), 3)

face_conv1 = cv2.resize(cv2.cvtColor(cv2.imread("conv0.png", -1), cv2.COLOR_BGR2GRAY), (144, 144))
face_conv2 = cv2.resize(cv2.cvtColor(cv2.imread("conv1.png", -1), cv2.COLOR_BGR2GRAY), (144, 144))
face_conv3 = cv2.resize(cv2.cvtColor(cv2.imread("conv2.png", -1), cv2.COLOR_BGR2GRAY), (144, 144))

background[0:144, 200:(200+144)] = face_conv1

background[(0+30):(144+30), (200+100):(200+100+144)] *= 0
background[(0+30):(144+30), (200+100):(200+100+144)] += 255

background[(0+30+3):(144+30+3), (200+100+3):(200+100+144+3)] = face_conv2

background[(0+30+3+30):(144+30+3+30), (200+100+3+100):(200+100+144+3+100)] *= 0
background[(0+30+3+30):(144+30+3+30), (200+100+3+100):(200+100+144+3+100)] += 255

background[(0+30+3+30+3):(144+30+3+30+3), (200+100+3+100+3):(200+100+144+3+100+3)] = face_conv3

background = cv2.arrowedLine(background, (565,108), (590,108), (0,0,0), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
background = cv2.putText(background, '"Surprise"', (605, 114), font, .65, (0,0,0), 1)

cv2.imwrite("dl_convolutions.png", background)
