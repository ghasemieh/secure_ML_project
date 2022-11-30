import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import keras.backend as K


class AdversarialDriving:
    def __init__(self, model, epsilon=1):
        self.model = model
        self.attack_type = None
        self.activate = False

        self.loss = K.mean(-self.model.output, axis=-1)
        self.grads = K.gradients(self.loss, self.model.input)

        # Get the sign of the gradient
        self.delta = K.sign(self.grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self.perturb = 0
        self.perturbs = []
        self.perturb_percent = 0
        self.perturb_percents = []
        self.n_attack = 1

        self.lr = 0.0002
        self.epsilon = epsilon
        self.xi = 4
        self.result = {}

    def init(self, attack_type, activate):

        # Reset Training Process
        if self.attack_type != attack_type:
            self.perturb = 0
            self.perturbs = []
            self.perturb_percent = 0
            self.perturb_percents = []
            self.n_attack = 1

        self.attack_type = attack_type

        if activate == 1:
            self.activate = True
            print("Attacker:", attack_type)
        else:
            self.activate = False
            print("No Attack")

            if attack_type == "image_specific_left":
                self.loss = -self.model.output
            if attack_type == "image_specific_right":
                self.loss = self.model.output

            self.grads = K.gradients(self.loss, self.model.input)
            # Get the sign of the gradient
            self.delta = K.sign(self.grads[0])

            print("Initialized", attack_type)

    def attack(self, input):
        if self.attack_type == "random":
            # Random Noises [-epsilon, +epsilon]
            noise = (np.random.randint(2, size=(160, 320, 3)) - 1) * self.epsilon
            return noise

        if self.attack_type.startswith("image_specific_"):
            noise = self.epsilon * self.sess.run(self.delta, feed_dict={self.model.input: np.array([input])})
            return noise.reshape(160, 320, 3)
