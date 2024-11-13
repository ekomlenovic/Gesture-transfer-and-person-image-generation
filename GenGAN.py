
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

        print(self.model)

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, 1).squeeze(1)

    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False,optSkeOrImage=1):
        self.optSkeOrImage = optSkeOrImage
        self.netD = Discriminator()
        self.real_label = 0.9
        self.fake_label = 0.1
        if optSkeOrImage==1:
            self.netG = GenNNSkeToImage()
            src_transform = None
            self.filename = 'data/DanceGenGANFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(64),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/DanceGenGANFromIm.pth'

        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename, map_location=device)  


    def train(self, n_epochs=20):
        criterion = nn.BCELoss()
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0004, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.00001, betas=(0.5, 0.999))

        for epoch in range(n_epochs):   
            for i, (ske, real_images) in enumerate(self.dataloader, 0):
                # Train Discriminator
                self.netD.zero_grad()
                real_images = real_images.to(device)
                # print("real_images.shape=", real_images.shape)
                # print("ske shape=", ske.shape)
                label = torch.full((real_images.size(0),), self.real_label, dtype=torch.float, device=device)
                output = self.netD(real_images)
                output = output.view(-1)  # Flatten the output to (batch_size,)
                lossD_real = criterion(output, label)
                lossD_real.backward()

                # Train with fake data
                if self.optSkeOrImage==2:
                    noise = torch.randn(ske.size(0), 3, 64, 64, device=device)
                else:
                    noise = torch.randn(real_images.size(0), 26, 1, 1, device=device)
                # print("noise.shape=", noise.shape)
                fake_images = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake_images.detach())
                output = output.view(-1)
                lossD_fake = criterion(output, label)
                lossD_fake.backward()

                optimizerD.step()
                lossD = lossD_real + lossD_fake

                # Train Generator
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake_images)
                output = output.view(-1)
                lossG = criterion(output, label)
                lossG.backward()

                optimizerG.step()

                print(f"Epoch [{epoch+1}/{n_epochs}], Step [{i}/{len(self.dataloader)}], "
                        f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}")
                

            # Save model periodically
            if epoch % 5 == 0:
                torch.save(self.netG, self.filename)




    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)
        ske_t_batch = ske_t_batch.to(device)
        normalized_output = self.netG(ske_t_batch)
        normalized_output = torch.Tensor.cpu(normalized_output)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False, optSkeOrImage)
        gen.train(1) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        # print("image.shape=", image.shape)
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

