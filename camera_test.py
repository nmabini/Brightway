import cv2
import sys
import time
from multiprocessing import Queue, Process
import numpy as np

class ImgHandler(Process):

    def __init__(self):
        self.imgQ = Queue()
        super(ImgHandler, self).__init__()

    def run(self):
        img_counter = 0
        while True:
            try:
                img = self.imgQ.get()
                if img:
                    #cv2.imshow("fromProcess", img)
                    frame = img[0]
                    #print(frame)
                    #cv2.imshow("fromProcess", frame)
                    #
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_blue = np.array([110,50,50])
                    upper_blue = np.array([130,255,255])
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    cv2.imshow('mask',mask)
                    #in ms
                    k = cv2.waitKey(1)
                    #cv2.destroyAllWindows()
                    if k%256 == 27:
                        #ESC
                        print("escape hit, closing...")
                        cv2.destroyAllWindows()
                        break
            except:
                pass
        

if __name__ == "__main__":
    imgProcess = ImgHandler()
    imgProcess.start()
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    while True:
        # Read picture. ret === True on success
        ret, frame = video_capture.read()
        '''
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            #ESC
            print("escape hit, closing...")
            break
        elif k%256 == 32:
            #print(frame)
            print(type(frame))
            imgProcess.imgQ.put([frame])
        '''
        #in ms
        k = cv2.waitKey(100)
        imgProcess.imgQ.put([frame])

    video_capture.release()
    cv2.destroyAllWindows()
