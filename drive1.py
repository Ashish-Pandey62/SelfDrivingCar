import asyncio
import websockets
import json  # Import JSON for parsing the data
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Load the model
model = load_model('model/model.h5')

speed_limit = 10

# Preprocess the image as before
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

async def handle_telemetry(websocket):
    async for message in websocket:
        try:
            # Print the raw message to debug
            print(f"Raw message: {message}")

            # Check if the message is a valid JSON object
            if message.startswith('{') and message.endswith('}'):
                # Parse the incoming message as JSON
                data = json.loads(message)  # This might throw an error if the message isn't valid JSON
                print("Telemetry data received")

                # Access speed and image data from the parsed dictionary
                speed = float(data['speed'])
                image = Image.open(BytesIO(base64.b64decode(data['image'])))
                image = np.asarray(image)
                image = img_preprocess(image)
                image = np.array([image])
                
                # Predict steering angle using the model
                steering_angle = float(model.predict(image))
                throttle = 1.0 - speed / speed_limit

                print(f"Steering: {steering_angle}, Throttle: {throttle}, Speed: {speed}")

                await send_control(websocket, steering_angle, throttle)
            else:
                print(f"Received a number instead of JSON. Message: {message}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")  # Catch JSON parsing errors
        except Exception as e:
            print(f"Error: {e}")

async def send_control(websocket, steering_angle, throttle):
    control_data = {
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }
    await websocket.send(json.dumps(control_data))  # Send as JSON

async def server(websocket, path):
    print("Connection established")
    await handle_telemetry(websocket)

# Start the server
start_server = websockets.serve(server, "0.0.0.0", 4567)

# Use asyncio to run the WebSocket server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
