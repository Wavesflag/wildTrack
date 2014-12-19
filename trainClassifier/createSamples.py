
import Tkinter
import tkMessageBox
import cv2
import cv
import numpy as np
from SimpleCV import Image, Display


cap = cv2.VideoCapture('/home/ctorney/data/wildebeest/from_grant/GOPR0076.MP4')
cap.set(cv.CV_CAP_PROP_POS_FRAMES,10010)
params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 1.0;
params.filterByInertia = False;
params.filterByConvexity = False;
params.filterByColor = False;
params.filterByCircularity = False;
params.filterByArea = True;
params.minArea = 5.0;
params.maxArea = 200.0;
params.minThreshold = 15;
params.maxThreshold = 255;

b = cv2.SimpleBlobDetector(params)

display = Display()
counter=0
box_dim = 48

while display.isNotDone():

# Capture frame-by-frame
    ret, frame = cap.read()
    blob = b.detect(frame)

    fcount=0
    for beest in blob:

        if fcount>100:
            continue
        tmpImg = Image(frame, cv2image=True).crop(int(beest.pt[0]),int(beest.pt[1]),box_dim,box_dim,centered=True)
        if ((tmpImg.width + tmpImg.height) == 2*box_dim):
#            cv2.imshow('classify',tmpImg.getNumpyCv2())
            tmpImg.save(display)
            result = tkMessageBox.askquestion("Wildebeest!", "Is this one? (no if you don't know)", icon='warning', type='yesnocancel')
            if result == 'yes':
                save_path = "yes/img-" + str(counter) + ".png"
                tmpImg.save(save_path)
                fcount += 1
                counter += 1
            if result == 'no':
                save_path = "no/img-" + str(counter) + ".png"
                tmpImg.save(save_path)
                counter += 1
            if result == 'cancel':
                display.done = True
                break



        # Display the resulting frame
#       cv2.imshow('frame',frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
display.quit()

