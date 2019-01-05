# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:20:25 2017

@author: pan
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from hashtable import hashtable


Qangle = 24
Qstrenth = 3
Qcoherence = 3

mat = cv2.imread("./thetrain/william-merritt-chase_shinnecock-hills-longisland.jpg")
h = np.load("lowR2.npy")
mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:,:,2]
LR = cv2.resize(mat,(0,0),fx=1.5,fy=1.5)
LRDirect = np.zeros((LR.shape[0],LR.shape[1]))
for xP in range(5,LR.shape[0]-6):
    for yP in range(5,LR.shape[1]-6):
        patch = LR[xP-5:xP+6,yP-5:yP+6]
        [angle,strenth,coherence] = hashtable(patch,Qangle,Qstrenth,Qcoherence)
        j = angle*9+strenth*3+coherence
        A = patch.reshape(1,-1)
        t = xP%2*2+yP%2
        hh = np.matrix(h[j,t])
        LRDirect[xP][yP] = hh*A.T
print("Test is off")



mat = cv2.imread("./thetrain/william-merritt-chase_shinnecock-hills-longisland.jpg")
#mat2 = cv2.imread("./test/1.jpg")
mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)
LR = cv2.resize(mat,(0,0),fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
LRDirectImage = LR
LRDirectImage[:,:,2] = LRDirect
A=cv2.cvtColor(LRDirectImage, cv2.COLOR_YCrCb2RGB);
im = Image.fromarray(A)
im.save('./hulu2.jpg')
#axes[1].imshow(cv2.cvtColor(LRDirectImage, cv2.COLOR_YCrCb2RGB))
#axes[1].set_title('RAISR')
#axes[2].imshow(cv2.cvtColor(mat2, cv2.COLOR_YCrCb2RGB))
#axes[2].set_title('MC-SRCNN')
#fig.savefig("./Baby2.jpg")