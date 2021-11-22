# import the necessary packages
from tensorflow.keras.models import load_model 
import tensorflow as tf
import numpy as np 
import dlib
import cv2 


# Load the model
model = load_model('./model/outputs/my_model.h5')

# Define the video stream
cap = cv2.VideoCapture(0)



def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the OpenCV format (x, y, w, h) 
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

classes = {0:'Correctly Masked Face', 1:'Incorrectly Masked Face', 2:'No Mask' }
colors = {0:(0, 255, 0), 1:(255, 0, 0), 2: (0, 0, 255) }


if not cap.isOpened():
    print("Cannot open camera!")
else:     
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # initialize dlib’s pre-trained face detector
        detector = dlib.get_frontal_face_detector()
        # load image in grayscale mode
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # detect faces in the grayscale image
        faces = detector(gray, 1)
        
        output = frame.copy()
        
        # show detected faces
        for rect in faces:
            (x, y, w, h) = rect_to_bb(rect)
            
            # extract region of interest 
            ROI = frame[x:x+w,y:y+h].copy()
            # resize image 
            ROI = cv2.resize(ROI, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            
            
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`
            input_tensor = tf.convert_to_tensor(ROI)
            # The model expects a batch of images, so add an axis with `tf.newaxis`
            input_tensor = input_tensor[tf.newaxis, ...]
            
            # make prediction
            y_hat = model.predict(input_tensor)    
            y_hat = np.argmax(y_hat ,axis=1)[0]
            text = classes[y_hat]
            color = colors[y_hat]
                                    
            output = cv2.rectangle(output, (x, y),(x+w, y+h), color, 2)
            cv2.putText(output, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
        # Display the resulting frame
        cv2.imshow('frame', output)
            
        if cv2.waitKey(1) == ord('q'):
            break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


