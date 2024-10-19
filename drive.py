import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sio = socketio.Server()
app = Flask(__name__)  # Use __name__ for the Flask app context
speed_limit = 10

def img_preprocess(img):
    try:
        logger.debug("Starting image preprocessing")
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255.0
        logger.debug("Image preprocessing completed successfully")
        return img
    except Exception as e:
        logger.error("Error in image preprocessing: %s", e)
        return None  # Return None if an error occurs

@sio.on('telemetry')
def telemetry(sid, data):
    logger.info("Telemetry data received")
    try:
        speed = float(data['speed'])
        logger.debug("Speed: %s", speed)

        # Decode and process the image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        image = np.asarray(image)
        
        processed_image = img_preprocess(image)
        if processed_image is None:
            logger.error("Image preprocessing failed, skipping this telemetry.")
            return
        
        processed_image = np.array([processed_image])
        steering_angle = float(model.predict(processed_image))
        throttle = 1.0 - speed / speed_limit
        
        logger.info("Steering angle: %s, Throttle: %s, Speed: %s", steering_angle, throttle, speed)
        send_control(steering_angle, throttle)
    except KeyError as e:
        logger.error("Missing key in telemetry data: %s", e)
    except ValueError as e:
        logger.error("Value error in telemetry data: %s", e)
    except Exception as e:
        logger.error("Error in telemetry handling: %s", e)

@sio.on('connect')
def connect(sid, environ):
    logger.info("Client connected: %s", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    try:
        logger.debug("Sending control commands")
        sio.emit('steer', data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        })
        logger.info("Control commands sent: steering_angle=%s, throttle=%s", steering_angle, throttle)
    except Exception as e:
        logger.error("Error sending control commands: %s", e)

if __name__ == '__main__':
    try:
        model = load_model('model/model.h5')
        logger.info("Model loaded successfully")
        app = socketio.Middleware(sio, app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except Exception as e:
        logger.error("Failed to start the server: %s", e)
