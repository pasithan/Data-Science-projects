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
model = load_model('face_mask_detection_model.h5')

# img1 = cv2.imread('test images/10.png')
# img2 = cv2.imread('test images/1008.png')
# img3 = cv2.imread('test images/1021.png')
# img4 = cv2.imread('test images/1030.png')
# img5 = cv2.imread('test images/1033.png')
# img6 = cv2.imread('test images/mask1.png')
# img7 = cv2.imread('test images/mask2.png')

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# resized1 = cv2.resize(img1, (256, 256)).reshape(1,256,256,3)
# resized2 = cv2.resize(img2, (256, 256)).reshape(1,256,256,3)
# resized3 = cv2.resize(img3, (256, 256)).reshape(1,256,256,3)
# resized4 = cv2.resize(img4, (256, 256)).reshape(1,256,256,3) 
# resized5 = cv2.resize(img5, (256, 256)).reshape(1,256,256,3) 
# resized6 = cv2.resize(img6, (256, 256)).reshape(1,256,256,3) 
# resized7 = cv2.resize(img7, (256, 256)).reshape(1,256,256,3) 

# pred1 = model.predict(resized1)
# pred2 = model.predict(resized2)
# pred3 = model.predict(resized3)
# pred4 = model.predict(resized4)
# pred5 = model.predict(resized5)
# pred6 = model.predict(resized6)
# pred7 = model.predict(resized7)

# print(pred1)
# print(pred2)
# print(pred3)
# print(pred4)
# print(pred5)
# print(pred6)
# print(pred7)


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


