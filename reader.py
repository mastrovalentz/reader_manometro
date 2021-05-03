import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
import os

#load image
im = Image.open("images/manometro_2.jpg")
plt.imshow(im)
plt.show()

#cricle cropping function
def crop_circle_jpg(m, cx, cy, r):
    m_crop = np.ones((2*r,2*r,m.shape[2]), dtype=np.int32)*255
    for i in range(2*r):
        for j in range(2*r):
            ii=cx-r+i
            jj=cy-r+j
            if np.sqrt(np.square(ii-cx)+np.square(jj-cy))<r:
                m_crop[i,j]=m[ii,jj]
    return m_crop

#read left sensor
def read_left(image, plot=False):
    #crop circle image
    im_left  = crop_circle_jpg(np.asarray(image), 270,270,150)
    #remove blocks(min and central square)
    im_left[120:200,70:240]=[255,255,255]
    im_left[250:,:]=[255,255,255]
    
    #set "white" threshold
    trsh = 180
    im_left_coords = np.where(np.mean(im_left,axis=2)<trsh)
    im_left_coords =np.array(im_left_coords)
        
    #first fit
    reg = np.polyfit(im_left_coords[0], im_left_coords[1], deg=1, full=True)
    fit =reg[0]
    res=reg[1][0]
    
    # Make predictions
    y_pred = im_left_coords[0]*fit[0]+fit[1]
    
    #l1 error for rejection
    l1_err = np.mean(np.abs(im_left_coords[1]-y_pred))
    #l1 1-sigma rejection
    im_left_rej = im_left_coords[:,np.abs(im_left_coords[1]-y_pred)<l1_err]
    #second fit
    reg = np.polyfit(im_left_rej[0], im_left_rej[1], deg=1, full=True)
    fit =reg[0]
    res=reg[1][0]
    y_pred = im_left_rej[0]*fit[0]+fit[1]
    #angle estimation 90+ since jpg system is rotated respect to pixel notation
    theta =90+np.arctan(fit[0])*180/np.pi
    if plot:
        plt.scatter(im_left_coords[0],im_left_coords[1], s=0.5)
        plt.scatter(im_left_rej[0],im_left_rej[1], s=0.5, c='green')
        plt.plot(im_left_rej[0], y_pred, c='red')
        plt.show()
        
        plt.imshow(im_left)
        plt.plot(y_pred,im_left_rej[0],  c='red')
        plt.show()
    return theta


#read right sensor
def read_right(image, plot=False):
    #crop circle image
    im_rigth = crop_circle_jpg(np.asarray(image), 260,910,200)
    #remove blocks(min and central square)
    im_rigth[180:250,100:280]=[255,255,255]
    im_rigth[110:140,45:120]=[255,255,255]
    im_rigth[105:140,245:350]=[255,255,255]
    
    #set "white" threshold
    trsh = 180
    im_right_coords = np.where(np.mean(im_rigth,axis=2)<trsh)
    im_right_coords =np.array(im_right_coords)
        
    #first fit
    reg = np.polyfit(im_right_coords[0], im_right_coords[1], deg=1, full=True)
    fit =reg[0]
    res=reg[1][0]
    
    # Make predictions
    y_pred = im_right_coords[0]*fit[0]+fit[1]
    
    #l1 error for rejection
    l1_err = np.mean(np.abs(im_right_coords[1]-y_pred))
    #l1 1-sigma rejection
    im_right_rej = im_right_coords[:,np.abs(im_right_coords[1]-y_pred)<l1_err]
    #second fit
    reg = np.polyfit(im_right_rej[0], im_right_rej[1], deg=1, full=True)
    fit =reg[0]
    res=reg[1][0]
    y_pred = im_right_rej[0]*fit[0]+fit[1]
    #angle estimation 90+ since jpg system is rotated respect to pixel notation
    theta =90+np.arctan(fit[0])*180/np.pi
    if plot:
        plt.scatter(im_right_coords[0],im_right_coords[1], s=0.5)
        plt.scatter(im_right_rej[0],im_right_rej[1], s=0.5, c='green')
        plt.plot(im_right_rej[0], y_pred, c='red')
        plt.show()
        
        plt.imshow(im_rigth)
        plt.plot(y_pred,im_right_rej[0],  c='red')
        plt.show()
    return theta

print("Left angle:", read_left(im, plot=True))
print("Right angle:", read_right(im, plot=True))

