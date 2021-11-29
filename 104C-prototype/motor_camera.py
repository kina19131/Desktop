import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# import the opencv library
import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    #low_red = np.array([264, 27, 100])
    #high_red = np.array([270, 65, 60])
    high_red = np.array([320,77,67])
    low_red = np.array([179, 255, 255])

    while ret == True:
        ret, frame = vid.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low_red, high_red)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
vid.release()
    




'''
dataset = https://www.tensorflow.org/datasets/catalog/malaria
image_picker = https://imagecolorpicker.com/
hex_to_hsv = https://colordesigner.io/convert/hextohsv
'''