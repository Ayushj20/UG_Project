#Importing libraries
import numpy as np
import cv2
#Reading image and videos
capture=cv2.VideoCapture(1)
image=cv2.imread('Image_to_Project.jpg')
video=cv2.VideoCapture('Sample.mp4')

detection=False
frameCounter=0
#read video and resize it 
success,targetvideo=video.read()
height,width,depth=image.shape
targetvideo=cv2.resize(targetvideo,(width,height))
#set orb nfeatures and keypoints
orb=cv2.ORB_create(nfeatures=1200)
keypoints1,descriptors1=orb.detectAndCompute(image,None)
#image=cv2.drawKeypoints(image,keypoints1,None)

#initialise while loop
while True:
    success,webcamimg=capture.read()
    augment=webcamimg.copy()
    
    #detecting the keypoints of web cam
    keypoints2,descriptors2=orb.detectAndCompute(webcamimg,None)
    #webcamimg=cv2.drawKeypoints(webcamimg,keypoints2,None)

    if detection==False:
        video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter==video.get(cv2.CAP_PROP_FRAME_COUNT):
          video.set(cv2.CAP_PROP_POS_FRAMES,0)
          frameCounter=0
        success,targetvideo=video.read()
        targetvideo=cv2.resize(targetvideo,(width,height))

    
    #creating a bruteforce Matcher
    bruteforce=cv2.BFMatcher()
    matches=bruteforce.knnMatch(descriptors1,descriptors2,k=2)
    #getting matchingpoints
    fittingpoints=[]
    for m,n in matches:
        if m.distance < .75*n.distance:
            fittingpoints.append(m)
    #print(len(fittingpoints))
    #drawing matchs
    imgmatching=cv2.drawMatches(image,keypoints1,webcamimg,keypoints2,fittingpoints,None,flags=2)
    

    if len(fittingpoints)>18:
        detection=True
        sourcepoints=np.float32([keypoints1[m.queryIdx].pt for m in fittingpoints]).reshape(-1,1,2)
        destinationpoints=np.float32([keypoints2[m.trainIdx].pt for m in fittingpoints]).reshape(-1,1,2)
        matrix,mask=cv2.findHomography(sourcepoints,destinationpoints,cv2.RANSAC,5)
        #print(matrix)

        points=np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
        destination=cv2.perspectiveTransform(points,matrix)
        image2=cv2.polylines(webcamimg,[np.int32(destination)],True,(255,0,0),2)

        imagewarp=cv2.warpPerspective(targetvideo,matrix,(webcamimg.shape[1],webcamimg.shape[0]))

        newmask=np.zeros((webcamimg.shape[0],webcamimg.shape[1]),np.uint8)
        cv2.fillPoly(newmask,[np.int32(destination)],(255,255,255))
        invert=cv2.bitwise_not(newmask)
        augment=cv2.bitwise_and(augment,augment,mask=invert)
        augment=cv2.bitwise_or(imagewarp,augment) 


    cv2.imshow('Augmented Video',augment)
    #cv2.imshow('Image2',image2)
    #cv2.imshow('Imagewarp',imagewarp)
    #cv2.imshow('ImgMatch',imgmatching)
    #cv2.imshow('Image',image)
    #cv2.imshow('targetvideo',targetvideo)
    #cv2.imshow('webcamimg',webcamimg)
    key=cv2.waitKey(1) 
    if key%256== 27 :
        break
    frameCounter+=1
