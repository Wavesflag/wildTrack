
import Tkinter
import tkMessageBox
import cv2
import cv
import numpy as np
from SimpleCV import Image, Display


cap = cv2.VideoCapture('/home/ctorney/data/wildebeest/from_grant/GOPR0076.MP4')
cap.set(cv.CV_CAP_PROP_POS_FRAMES,10010)

display = Display()
counter=0
box_dim = 48

ret, frame = cap.read()
while display.isNotDone():

# Capture frame-by-frame


    thisIm = Image(frame, cv2image=True)
    thisIm.save(display)
    if display.mouseLeft:
        tmpImg = thisIm.crop(display.mouseX, display.mouseY, box_dim,box_dim, centered=True)
        save_path = "no/img-n-" + str(counter) + ".png"
        tmpImg.save(save_path)
        counter += 1
        thisIm.save(display)
        display.mouseLeft = False
    if display.mouseRight:
        ret, frame = cap.read()



        # Display the resulting frame
#       cv2.imshow('frame',frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
display.quit()

