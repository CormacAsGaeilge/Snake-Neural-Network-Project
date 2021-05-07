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

lr = 1e-2
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
pygame.display.set_caption('Snake AI')

clock = pygame.time.Clock()

snake_block = 10
snake_speed = 20

font_style = pygame.font.SysFont("bahnschrift", 15)
score_font = pygame.font.SysFont("comicsansms", 25)

maxNoGames = 20
currentGame = 1
moves = []


def randomMove():
    return round(random.randrange(-1, 2))


def Your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])


def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])


def handleGameOver(Length_of_snake, game_over, game_close, currentGame, moves):
    dis.fill(blue)
    message("You Lost! Press C-Play Again or Q-Quit", red)
    Your_score(Length_of_snake - 1)
    pygame.display.update()

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
        print("ROTATION: ", previousPosition)
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


def gameLoop(currentGame, moves):
    while maxNoGames > currentGame:
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
        while not game_over:
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
                    Length_of_snake, game_over, game_close, currentGame, moves)

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
                if (x[0]+1) == snake_Head[0] and x[1] == snake_Head[1]:
                    isLeftBlocked = 1
                if (x[0]-1) == snake_Head[0] and x[1] == snake_Head[1]:
                    isRightBlocked = 1
                if x[0] == snake_Head[0] and (x[1]+1) == snake_Head[1]:
                    isUpBlocked = 1
                if x[0] == snake_Head[0] and (x[1]-1) == snake_Head[1]:
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

            prev_observation = []
            suggestedTurn = randomMove()
            if len(snake_List) > 0:
                prev_observation = generate_observation(
                    lBlocked, fBlocked, rBlocked, snake_List, [foodx, foody], previousPosition)
                prev_distance = get_food_distance(snake_List, [foodx, foody])
                predictions = []
                for action in range(-1, 2):
                    predictions.append(nn_model.predict(add_action_to_observation(
                        action, prev_observation).reshape(-1, 5, 1)))
                suggestedTurn = np.argmax(np.array(predictions)) - 1
                print("PREDICTIONS: ", predictions)
                print("Suggested TURN: ", suggestedTurn)
                print("PREVIOUS DIRECTION: ", previousDirection)

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
                output = -15

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
                    output = -15

            our_snake(snake_block, snake_List)
            Your_score(Length_of_snake - 1)

            pygame.display.update()

            if x1 == foodx and y1 == foody:
                foodx, foody = placeFruit()
                Length_of_snake += 1
                output = 1

            clock.tick(snake_speed)
            # moves.append(''.join(map(str,[isLeftBlocked, ',', isRightBlocked, ',', isUpBlocked, ',', isDownBlocked, ',', suggestedDirection, '\n'])))
            if len(prev_observation) == 4 and not game_over:
                if get_food_distance(snake_List, [foodx, foody]) < prev_distance:
                    output = 1
                #     print("CLOSER!")
                # else:
                #     print("FURTHER!")
                print([add_action_to_observation(
                    suggestedTurn, prev_observation), output])
    dt = np.dtype([])
    # np.savetxt("trainingData.csv", np.array(moves, dtype=dt), delimiter=",")
    # training_file = open("trainingData.txt","w+")
    # training_file.write(str(moves))
    # training_file.close()
    pygame.quit()
    # quit()


def model():
    network = tflearn.layers.core.input_data(shape=[None, 5, 1], name='input')
    network = tflearn.layers.core.fully_connected(
        network, neurons, activation='relu')
    network = tflearn.layers.core.fully_connected(
        network, 1, activation='linear')
    network = tflearn.layers.estimator.regression(
        network, optimizer='adam', learning_rate=lr, loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


nn_model = model()
nn_model.load(filename)
gameLoop(currentGame, moves)
