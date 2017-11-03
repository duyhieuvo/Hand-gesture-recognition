import cv2
import numpy as np

# initialize:
frame = None #the current frame of the video
pts = np.zeros(shape=(8,3), dtype= np.uint8) #the list of color sample points of the hand
inputMode = False #whether or not this is input mode
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
counter = 0 #count the number of samples which have been selected
#clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
#fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 500, detectShadows = 0) #create background subtractor

'''
def reduceLightEffect(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(frame)
    cv = cv2.equalizeHist(v)
    frame = cv2.merge((h,s,cv))
    frame = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
    return frame
'''
def nothing(x):
    pass

def adjust():
    img = np.zeros((300,512), np.uint8)
    cv2.namedWindow('Adjustment')
    # create trackbars for H,S,V variation
    cv2.createTrackbar('H','Adjustment',0,179,nothing)
    cv2.createTrackbar('S','Adjustment',0,255,nothing)
    cv2.createTrackbar('V','Adjustment',0,255,nothing)

def selectSample(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, pts, inputMode, counter
    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and counter < 8:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        pts[counter,0] = h[y,x]
        pts[counter,1] = s[y,x]
        pts[counter,2] = v[y,x]
        counter+=1
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def constructImage(frame,pts):
    h = cv2.getTrackbarPos('H', 'Adjustment')
    s = cv2.getTrackbarPos('S', 'Adjustment')
    v = cv2.getTrackbarPos('V', 'Adjustment')
    temp = np.zeros(frame.shape[:2],dtype="uint8")
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    for i in range(8):
        if (pts[i,0] - h) < 0:
            lh = 0
        else:
            lh = pts[i,0] - h
        if (pts[i,1] - s) < 0:
            ls = 0
        else:
            ls = pts[i,1] - s
        if (pts[i,2] - v) < 0:
            lv = 0
        else:
            lv = pts[i,2] - v
        if (pts[i,0] + h) > 179:
            hh = 179
        else:
            hh = pts[i,0] + h
        if (pts[i,1] + s) > 255:
            hs = 255
        else:
            hs = pts[i,1] + s
        if (pts[i,2] + v) > 255:
            hv = 255
        else:
            hv = pts[i,2] + v
        lower = np.array([lh,ls,lv])
        upper = np.array([hh,hs,hv])
        mask = cv2.inRange(hsv,lower,upper)
        temp = cv2.add(mask,temp)

    temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    #temp = cv2.erode(temp,kernel)
    #temp = cv2.dilate(temp, kernel)
    binaryImage = temp #cv2.medianBlur(temp,5)
    return binaryImage

'''
def backgroundSubtractor(frame):
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,small_kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    img = cv2.bitwise_and(frame,frame,mask = fgmask)
    return img
'''

def findContour(frame,img):
    (image, cnts, hierarchy) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
    cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
    print len(cnts)
    cnt = cnts[0]
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(frame,start,end,[255,0,0],2)
        cv2.circle(frame,far,5,[0,0,255],-1)


def main():
    global frame, pts, inputMode, counter
    adjust()
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectSample)
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # check to see if we have reached the end of the video
        if not grabbed:
            break
        #frame = reduceLightEffect(frame)
        if counter == 8:
            binaryImage = constructImage(frame,pts)
            #frame = cv2.bitwise_and(frame,frame,mask = binaryImage)
            #frame = backgroundSubtractor(frame)
            findContour(frame,binaryImage)
            cv2.imshow("binary image",binaryImage)
        # show the frame and record if the user presses a key
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #h = frame[:,:,0]
        #cv2.imshow("frame", h)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("i") and counter < 8:
            # indicate that we are in input mode and clone the frame
            inputMode = True
            while counter < 8:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)
                constructImage(frame,pts)
        elif key == ord("q"):
            break
        # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
