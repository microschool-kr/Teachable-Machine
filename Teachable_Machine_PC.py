import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Set WebCam
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth) # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight) # Height

# Load the model
model = load_model('keras_model.h5')

while True:
  # take picture
  success, img = cap.read()
  cv2.imshow("", img)
  
  # data preprocessing
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  size = (224, 224)
  
  # OpenCV to PIL image
  # convert from BGR to RGB & from openCV2 to PIL
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img=Image.fromarray(img)
  
  # PIL image processiing
  image = ImageOps.fit(img, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  normalized_image_array = (image_array.astype(np.float32) / 127.) - 1
  data[0] = normalized_image_array
  
  # predicting
  prediction = model.predict(data)
  print("Pen", prediction[0,0])
  print("Key", prediction[0,1])
  print("Vaseline", prediction[0,2])
  
  # expiration
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
