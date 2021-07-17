# import the opencv library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json
from keras.preprocessing.image import load_img
import tensorflow as tf

# Drow rounding box
def draw_box(frame, pred, conf, loc):
	font = cv2.FONT_HERSHEY_SIMPLEX
	(x, y, w, h) = loc
	if pred == 0: 
		print('You are wearing mask with {} confidence'.format(conf))
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)	# Draw square
		cv2.putText(frame, 'Mask ({:.2f}%)'.format(conf[0]*100), (x+w+5, y+h+5), font, 1, (0,255,0), 3)	
	else: 
		print('You are not wearing mask with {} confidence'.format(conf))
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)	# Draw square
		cv2.putText(frame, 'No Mask ({:.2f}%)'.format(conf[0]*100), (x+w+5, y+h+5), font, 1, (255,0,0), 3)	

	return frame


# Load face mask detection model
model = load_model('face_mask_detection_model_aug.h5')

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

	# Convert bgr to rgb
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to gray color
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
	faces = face_cascade.detectMultiScale(gray, 2, 3)

	# Define variale for keeping roi faces
	cropped_faces = []
	roi = None

    # Draw rectangle around the faces
	for (x, y, w, h) in faces:
		roi = frame[y: y+h, x: x+w].copy()	# Crop roi
		resized_face = cv2.resize(roi, (256, 256)).reshape(1,256,256,3)	# Resize image
		pred = model.predict(resized_face)	# Predict the cropped image
		conf = np.max(pred, axis=-1)	# Calculate confidence 
		pred = np.argmax(pred, axis= -1)	
		cv2.imshow('cropped', cv2.cvtColor(resized_face.reshape(256, 256, 3), cv2.COLOR_RGB2BGR))
		frame = draw_box(frame, pred, conf, (x, y, w, h))
		
	# Display the resulting frame
	cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


