from random import randint
import numpy as np
import tflearn
import math
import tflearn.layers.core
import tflearn.layers.estimator
from statistics import mean
import time

lr = 0.0005
batch_size = 8
neurons = 250
epochs = 3
filename = 'snake_nn.tflearn'


def generate_observation(isLeftBlocked, isRightBlocked, isUpBlocked, isDownBlocked, snake_List, food):
    food_direction = get_food_direction_vector(snake_List, food)
    snake_direction = get_snake_direction_vector(snake_List)

    angle = get_angle(snake_direction, food_direction)
    # print("Angle: ", str(angle))
    return np.array([isLeftBlocked, isRightBlocked, isUpBlocked, isDownBlocked, angle])


def get_snake_direction_vector(snake_List):
    if len(snake_List) > 1:
        return np.array(snake_List[0]) - np.array(snake_List[1])
    return np.array([0, 1])


def get_food_direction_vector(snake, food):
    return np.array(food) - np.array(snake[0])


def get_angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def add_action_to_observation(action, observation):
    return np.append([action], observation)


def get_food_distance(snake, food):
    return np.linalg.norm(get_food_direction_vector(snake, food))


def model():
    network = tflearn.layers.core.input_data(shape=[None, 6, 1], name='input')
    network = tflearn.layers.core.fully_connected(
        network, neurons, activation='relu')
    network = tflearn.layers.core.fully_connected(
        network, 1, activation='linear')
    network = tflearn.layers.estimator.regression(
        network, optimizer='adam', learning_rate=lr, batch_size=batch_size, loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


def train_model(training_data, model):
    print('--- train_model ---')
    start = time.time()
    x = np.array([i[0] for i in training_data]).reshape(-1, 6, 1)
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model.fit(x, y, n_epoch=epochs, shuffle=True, run_id=filename)
    model.save(filename)
    end = time.time()
    print(time.strftime("Time elapsed: %H:%M:%S", time.gmtime(end - start)))
    return model

nn_model = model()
nn_model.load(filename)

snake_List = []
snake_Head = []
snake_Head.append(5)
snake_Head.append(7)
snake_List.append(snake_Head)
prev_observation = generate_observation(0, 0, 0, 1, snake_List, [6, 3])

predictions = []
for action in range(0, 4):
    predictions.append(nn_model.predict(add_action_to_observation(
        action, prev_observation).reshape(-1, 6, 1)))
action = np.argmax(np.array(predictions))

print(predictions)

print("Chosen")

print(action)
