import base64
import os
from io import BytesIO

import cv2
import pandas as pd
import tensorflow as tf

import configuration
from configuration import ConfigParser

tf.compat.v1.disable_eager_execution()
import keras.backend as K
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from colorama import Fore
from flask import Flask
from flask_cors import CORS
from keras.models import load_model

sio = socketio.Server(cors_allowed_origins='*')

app = Flask(__name__)
CORS(app)
MAX_SPEED = 20
MIN_SPEED = 10
global counter
counter = 0
EPSILON = 1
APPLY_ATTACK = True
speed_limit = MAX_SPEED
file_path = "record.csv"


class myAttacker:
    def __init__(self, model, activate, attack_type, epsilon=1):
        self.model = model
        self.attack_type = attack_type
        self.activate = activate
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.epsilon = epsilon
        if activate == 1:
            self.activate = True
        else:
            self.activate = False
        self.attack_type = attack_type
        if attack_type == "turn_left":
            self.loss = -self.model.output
        if attack_type == "turn_right":
            self.loss = self.model.output
        self.grads = K.gradients(self.loss, self.model.input)
        self.delta = K.sign(self.grads[0])

    def run(self, input):
        if self.attack_type == "random":
            noise = (np.random.randint(2, size=(160, 320, 3)) - 1) * self.epsilon
            return noise

        if self.attack_type.startswith("turn_"):
            noise = self.epsilon * self.sess.run(self.delta, feed_dict={self.model.input: np.array([input])})
            return noise.reshape(160, 320, 3)


def img2base64(image):
    origin_img = Image.fromarray(np.uint8(image))
    origin_buff = BytesIO()
    origin_img.save(origin_buff, format="JPEG")
    return base64.b64encode(origin_buff.getvalue()).decode("utf-8")


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def preprocess(image):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (320, 160), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data["speed"])
        sio.emit('update', {'data': data["image"], 'speed': data["speed"]})
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = preprocess(image)
        sio.emit('input', {'data': img2base64(image)})
        y_true = model.predict(np.array([image]), batch_size=1)
        if adv_drv.activate:
            perturb = adv_drv.run(image)
            x_adv = np.array(image) + perturb
            sio.emit('adv', {'data': img2base64(x_adv)})
            sio.emit('diff', {'data': img2base64(perturb)})
            y_adv = float(model.predict(np.array([x_adv]), batch_size=1))
            sio.emit('res', {'original': str(float(y_true)), 'result': str(float(y_adv)),
                             'percentage': str(float(((y_true - y_adv) * 100 / np.abs(y_true))))})
            df = pd.DataFrame(data=[[y_true[0][0], y_adv, y_adv - y_true[0][0]]],
                              columns=["y_true", "y_adv", "y_diff"])
            try:
                if os.path.isfile(file_path):
                    df.to_csv(file_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(file_path, header=True, index=False)
            except:
                print("Close csv file")

            if APPLY_ATTACK:
                image = np.array([x_adv])
            else:
                image = np.array([image])

            telemetry.count = telemetry.count + 1
        else:
            image = np.array([image])
        steering_angle = float(model.predict(image, batch_size=1))
        global speed_limit, counter
        if speed > speed_limit:
            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED
        throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
        counter += 1
        if counter % 1 == 0:
            if not adv_drv.activate:
                print(Fore.WHITE + 'Steering angle: {}, Throttle: {}, Speed: {}'.format(round(steering_angle, 3),
                                                                                        round(throttle, 2),
                                                                                        round(speed, 1)))
            elif adv_drv.attack_type == 'turn_left':
                print(Fore.RED + 'Steering angle: {}, Throttle: {}, Speed: {}'.format(round(steering_angle, 3),
                                                                                      round(throttle, 2),
                                                                                      round(speed, 1)))
            elif adv_drv.attack_type == 'turn_right':
                print(Fore.GREEN + 'Steering angle: {}, Throttle: {}, Speed: {}'.format(round(steering_angle, 3),
                                                                                        round(throttle, 2),
                                                                                        round(speed, 1)))
            elif adv_drv.attack_type == 'random':
                print(Fore.YELLOW + 'Steering angle: {}, Throttle: {}, Speed: {}'.format(round(steering_angle, 3),
                                                                                         round(throttle, 2),
                                                                                         round(speed, 1)))
        send_control(steering_angle, throttle)

    else:
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()}, skip_sid=True)


if __name__ == '__main__':
    config: ConfigParser = configuration.get()
    telemetry.count = 0
    model = load_model('model.h5')
    model.summary()
    attack = config['Attack']['active']
    attack_type = config['Attack']['type']
    adv_drv = myAttacker(model, activate=int(attack), attack_type=attack_type, epsilon=EPSILON)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
