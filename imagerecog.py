import numpy as np
import imutils
import cv2, time # delete first # to use time module


# car detection
# idea: detect cars within a section of the image


#basic video capture
cap = cv2.VideoCapture(0)



while True:
    check, frame = cap.read()
    cv2.imshow('frame', frame)
    img = cv2.imread('headlights.jpg')

    #define range to look for
    lower_white = np.array([240, 240, 240])
    upper_white = np.array([255, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsvstill = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # only get white light
    mask = cv2.inRange(frame, lower_white, upper_white)
    mask2 = cv2.inRange(img, lower_white, upper_white)
    final = cv2.bitwise_and(frame, frame, mask=mask)
    final2 = cv2.bitwise_and(img, img, mask=mask2)

#    cv2.imshow('HSV', hsv)
    cv2.imshow('Headlights', img)
    cv2.imshow('Final', final)
    cv2.imshow('Headlight Final', final2)
    # quits when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
