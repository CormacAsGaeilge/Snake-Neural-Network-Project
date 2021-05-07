import pygame
import random
from random import randint
import numpy as np
import tflearn
import math
import tflearn.layers.core
import tflearn.layers.estimator
from statistics import mean
import time
from datetime import datetime


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr = 0.001
# batch_size = 8
neurons = 25
epochs = 3
filename = 'snake_nn.tflearn'

pygame.init()

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis_width = 200
dis_height = 200

dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake AI Training')

clock = pygame.time.Clock()

snake_block = 10
snake_speed = 5000000

font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

maxNoGames = 1000
maxSteps = 500
currentGame = 0
moves = []

averageScore = 0


def randomMove():
    return round(random.randrange(-1, 2))


def is_valid_cell(snake, cell, previousCell):
    # if cell[0] == previousCell[0] and cell[1] == previousCell[1]:
    #     print("Previously Here")
    #     return False
    # print("Current Position: ", snake[0])
    # print(snake[:-1])
    if cell in snake[:-1]:
        # print("Body Here", cell)
        return False
    elif cell[0] >= dis_width or cell[0] < 0 or cell[1] >= dis_height or cell[1] < 0:
        # print("Edge Here")
        return False
    else:
        return True


def smartMove(snake, foodx, foody, previousDirection):
    if len(snake) == 0:
        return randomMove()

    head = snake[-1]
    cells = []
    cell_left = [head[0] - snake_block, head[1]]
    cell_right = [head[0] + snake_block, head[1]]
    cell_up = [head[0], head[1] - snake_block]
    cell_down = [head[0], head[1] + snake_block]
    prevCell = [-1, -1]
    if previousDirection == 1:
        prevCell = cell_left
    if previousDirection == 0:
        prevCell = cell_right
    if previousDirection == 3:
        prevCell = cell_up
    if previousDirection == 2:
        prevCell = cell_down

    if previousDirection == 0:  # LEFT
        l_cell = cell_down
        f_cell = cell_left
        r_cell = cell_up
    if previousDirection == 1:  # RIGHT
        l_cell = cell_up
        f_cell = cell_right
        r_cell = cell_down
    if previousDirection == 2:  # UP
        l_cell = cell_left
        f_cell = cell_up
        r_cell = cell_right
    if previousDirection == 3:  # DOWN
        l_cell = cell_right
        f_cell = cell_down
        r_cell = cell_left

    # print("L:", cell_left)
    # print("R:", cell_right)
    # print("U:", cell_down)
    # print("D:", cell_up)
    # print("S:", snake)
    # print("F:", [foodx, foody])

    for cell in [l_cell, f_cell, r_cell]:
        if is_valid_cell(snake, cell, prevCell):
            food_distance = get_food_distance([cell], [foodx, foody])
            cells.append([cell, food_distance])
            cells.sort(key=lambda x: x[1])
        # else:
        #     print("Invalid: ", cell)
    # print(cells)
    if len(cells) == 0:
        # print("blocked")
        return 0  # snake is trapped, no way out. game over with next step
    # print("CELLS: ", cells)
    dir = np.array(cells[0][0]) - np.array(head)
    if cells[0][0] == l_cell:
        # print("left")
        return -1
    if cells[0][0] == f_cell:
        # print("forward")
        return 0
    if cells[0][0] == r_cell:
        # print("right")
        return 1
    # if cells[0][0] == cell_left:
    #     # print("left")
    #     return 0
    # if cells[0][0] == cell_right:
    #     # print("right")
    #     return 1
    # if cells[0][0] == cell_up:
    #     # print("down")
    #     return 2
    # if cells[0][0] == cell_down:
    #     # print("up")
    #     return 3


def Your_score(score, currentGame):
    value = score_font.render(
        "Game: "+str(currentGame)+" - Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])


def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])


def handleGameOver(Length_of_snake, game_over, game_close, currentGame, averageScore):
    dis.fill(blue)
    message("You Lost! Press C-Play Again or Q-Quit", red)
    Your_score(Length_of_snake - 1, currentGame)
    pygame.display.update()
    averageScore = averageScore + Length_of_snake - 1

    # for event in pygame.event.get():
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_q:
    #             game_over = True
    #             game_close = False
    #         if event.key == pygame.K_c:
    #             gameLoop()

    game_over = True
    game_close = False

    return game_over, game_close


def placeFruit():
    return round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0, round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0


def generate_observation(isLeftBlocked, isForwardBlocked, isRightBlocked, snake_List, food, previousPosition):
    food_direction = get_food_direction_vector(snake_List, food)
    snake_direction = get_snake_direction_vector(snake_List, previousPosition)

    angle = get_angle(snake_direction, food_direction)
    # print("Angle: ", str(angle))
    return np.array([isLeftBlocked, isForwardBlocked, isRightBlocked, angle])


def get_snake_direction_vector(snake_List, previousPosition):
    if len(snake_List) > 1:
        return np.array(snake_List[-1]) - np.array(snake_List[-2])
    else:
        # print("ROTATION: ", previousPosition)
        return np.array(snake_List[-1]) - np.array(previousPosition)



def get_food_direction_vector(snake, food):
    return np.array(food) - np.array(snake[-1])


def get_angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi


def normalize_vector(vector):
    normalized = np.linalg.norm(vector)
    if normalized:
        return vector / normalized
    return [0, 0]


def add_action_to_observation(action, observation):
    return np.append([action], observation)


def get_food_distance(snake, food):
    return np.linalg.norm(get_food_direction_vector(snake, food))


def gameLoop(currentGame, moves, averageScore):
    while maxNoGames > currentGame:
        step = 0
        currentGame = currentGame + 1
        game_over = False
        game_close = False
        x1 = dis_width / 2
        y1 = dis_height / 2

        x1_change = 0
        y1_change = 0

        snake_List = []
        Length_of_snake = 1

        foodx, foody = placeFruit()
        previousDirection = 1
        previousPosition = [x1+snake_block, y1]
        while not game_over and maxSteps > step:
            step = step + 1
            isLeftBlocked = 0
            isRightBlocked = 0
            isUpBlocked = 0
            isDownBlocked = 0
            suggestedDirection = 0
            output = 0

            lBlocked = 0
            rBlocked = 0
            fBlocked = 0

            while game_close == True:
                game_over, game_close = handleGameOver(
                    Length_of_snake, game_over, game_close, currentGame, averageScore)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True

            # Checking to see if adjecent squares are edge pieces
            if x1+1 >= dis_width:
                isRightBlocked = 1
            if x1-1 < 0:
                isLeftBlocked = 1
            if y1+1 >= dis_height:
                isDownBlocked = 1
            if y1-1 < 0:
                isUpBlocked = 1
            for x in snake_List[:-1]:
                if (x[0]+snake_block) == snake_Head[0] and x[1] == snake_Head[1]:
                    isLeftBlocked = 1
                if (x[0]-snake_block) == snake_Head[0] and x[1] == snake_Head[1]:
                    isRightBlocked = 1
                if x[0] == snake_Head[0] and (x[1]+snake_block) == snake_Head[1]:
                    isUpBlocked = 1
                if x[0] == snake_Head[0] and (x[1]-snake_block) == snake_Head[1]:
                    isDownBlocked = 1

            if previousDirection == 0:  # LEFT
                lBlocked = isDownBlocked
                fBlocked = isLeftBlocked
                rBlocked = isUpBlocked
                if len(snake_List) > 0:
                    previousPosition = [snake_Head[0] + snake_block, snake_Head[1]]
            if previousDirection == 1:  # RIGHT
                lBlocked = isUpBlocked
                fBlocked = isRightBlocked
                rBlocked = isDownBlocked
                if len(snake_List) > 0:
                    previousPosition = [snake_Head[0]-snake_block, snake_Head[1]]
            if previousDirection == 2:  # UP
                lBlocked = isLeftBlocked
                fBlocked = isUpBlocked
                rBlocked = isRightBlocked
                if len(snake_List) > 0:
                    previousPosition = [snake_Head[0], snake_Head[1]-snake_block]
            if previousDirection == 3:  # DOWN
                lBlocked = isRightBlocked
                fBlocked = isDownBlocked
                rBlocked = isLeftBlocked
                if len(snake_List) > 0:
                    previousPosition = [snake_Head[0], snake_Head[1]+snake_block]
                    
            # print("P: ", previousDirection, " L: ", lBlocked,
            #       " F: ", fBlocked, " R: ", rBlocked)
            # print("S: ", snake_List, " FD: ", [foodx, foody])
            prev_observation = []
            if len(snake_List) > 0:
                prev_observation = generate_observation(
                    lBlocked, fBlocked, rBlocked, snake_List, [foodx, foody], previousPosition)
                prev_distance = get_food_distance(snake_List, [foodx, foody])
            # if Length_of_snake < 7:
            #     suggestedTurn = smartMove(snake_List, foodx, foody, previousDirection)  
            # else:
            suggestedTurn = randomMove()

            if suggestedTurn == 0:  # FORWARD -> CONTINUE
                suggestedDirection = previousDirection

            if previousDirection == 0:  # LEFT
                if suggestedTurn == -1:  # LEFT -> LEFT is DOWN
                    suggestedDirection = 3
                if suggestedTurn == 1:  # LEFT -> RIGHT is UP
                    suggestedDirection = 2
            elif previousDirection == 1:  # RIGHT
                if suggestedTurn == -1:  # RIGHT -> LEFT is UP
                    suggestedDirection = 2
                if suggestedTurn == 1:  # RIGHT -> RIGHT is DOWN
                    suggestedDirection = 3
            elif previousDirection == 2:  # UP
                if suggestedTurn == -1:  # UP -> LEFT is LEFT
                    suggestedDirection = 0
                if suggestedTurn == 1:  # UP -> RIGHT is RIGHT
                    suggestedDirection = 1
            elif previousDirection == 3:  # DOWN
                if suggestedTurn == -1:  # DOWN -> LEFT is RIGHT
                    suggestedDirection = 1
                if suggestedTurn == 1:  # DOWN -> RIGHT is LEFT
                    suggestedDirection = 0

            # suggestedDirection = smartMove(snake_List, foodx, foody, previousDirection)
            previousDirection = suggestedDirection
            if suggestedDirection == 0:
                x1_change = -snake_block
                y1_change = 0
            elif suggestedDirection == 1:
                x1_change = snake_block
                y1_change = 0
            elif suggestedDirection == 2:
                y1_change = -snake_block
                x1_change = 0
            elif suggestedDirection == 3:
                y1_change = snake_block
                x1_change = 0

            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True
                output = -1
            # print("C:",[x1, y1])
            x1 += x1_change
            y1 += y1_change
            dis.fill(blue)
            pygame.draw.rect(
                dis, green, [foodx, foody, snake_block, snake_block])
            snake_Head = []
            snake_Head.append(x1)
            snake_Head.append(y1)
            snake_List.append(snake_Head)
            if len(snake_List) > Length_of_snake:
                del snake_List[0]

            for x in snake_List[:-1]:
                if x == snake_Head:
                    game_close = True
                    output = -1

            our_snake(snake_block, snake_List)
            Your_score(Length_of_snake - 1, currentGame)

            pygame.display.update()

            if x1 == foodx and y1 == foody:
                foodx, foody = placeFruit()
                Length_of_snake += 1

            clock.tick(snake_speed)
            # moves.append(''.join(map(str,[isLeftBlocked, ',', isRightBlocked, ',', isUpBlocked, ',', isDownBlocked, ',', suggestedDirection, '\n'])))
            if len(prev_observation) == 4 and not game_over:
                if get_food_distance(snake_List, [foodx, foody]) < prev_distance:
                    output = 1
                moves.append([add_action_to_observation(
                    suggestedTurn, prev_observation), output])
                # print(moves[-1])

    # np.savetxt("trainingData.txt", moves, delimiter=",")
    # training_file = open("trainingData.txt","w+")
    # training_file.write(str(moves))
    # training_file.close()
    pygame.quit()
    averageScore = averageScore/maxNoGames
    return averageScore
    # quit()


def model():
    network = tflearn.layers.core.input_data(shape=[None, 5, 1], name='input')
    network = tflearn.layers.core.fully_connected(
        network, neurons, activation='relu')
    network = tflearn.layers.core.fully_connected(
        network, 1, activation='linear')
    network = tflearn.layers.estimator.regression(
        network, optimizer='adam', learning_rate=lr, loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=3)
    return model


def train_model(training_data, model):
    now = datetime.now()

    timestamp = datetime.timestamp(now)
    print('--- train_model ---')
    start = time.time()
    x = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    model.fit(x, y, n_epoch=epochs, shuffle=True, run_id=str(timestamp)+filename)
    model.save(filename)
    end = time.time()
    print(time.strftime("Time elapsed: %H:%M:%S", time.gmtime(end - start)))
    return model


averageScore = gameLoop(currentGame, moves, averageScore)

training_data = moves
nn_model = model()
nn_model = train_model(training_data, nn_model)

snake_List = []
snake_Head = []
snake_Head.append(5)
snake_Head.append(7)
snake_List.append(snake_Head)
prev_observation = generate_observation(0, 0, 1, snake_List, [6, 3], [4, 7])

predictions = []
for action in range(-1, 2):
    predictions.append(nn_model.predict(add_action_to_observation(
        action, prev_observation).reshape(-1, 5, 1)))
action = np.argmax(np.array(predictions)) - 1

print(predictions)

print("Chosen")

print(action)


print("Average Score From Search Algorithm: ", str(averageScore))
