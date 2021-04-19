import pygame
import time
import random
 
pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 500
dis_height = 500
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Edureka')
 
clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 15000
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

maxNoGames = 20
currentGame = 1
moves = []

def randomMove():
    return round(random.randrange(0,4))


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
    
    if maxNoGames > currentGame:
        currentGame = currentGame + 1
        gameLoop(currentGame, moves)
    else:
        game_over = True
        game_close = False
        
    return game_over, game_close
 
def placeFruit():
    return round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0, round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

def gameLoop(currentGame, moves):
    game_over = False
    game_close = False
 
    x1 = dis_width / 2
    y1 = dis_height / 2
 
    x1_change = 0
    y1_change = 0
 
    snake_List = []
    Length_of_snake = 1
 
    foodx, foody = placeFruit()
    features = []
    while not game_over:
        isLeftBlocked = 0 
        isRightBlocked = 0 
        isUpBlocked = 0 
        isDownBlocked = 0
        suggestedDirection = 0
 
        while game_close == True:
            game_over, game_close = handleGameOver(Length_of_snake, game_over, game_close, currentGame, moves)
 
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         game_over = True
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             x1_change = -snake_block
        #             y1_change = 0
        #         elif event.key == pygame.K_RIGHT:
        #             x1_change = snake_block
        #             y1_change = 0
        #         elif event.key == pygame.K_UP:
        #             y1_change = -snake_block
        #             x1_change = 0
        #         elif event.key == pygame.K_DOWN:
        #             y1_change = snake_block
        #             x1_change = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                
        suggestedDirection = randomMove()
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
        
        # Checking to see if adjecent squares are edge pieces 
        if x1+1 >= dis_width:
            isLeftBlocked = 1
            
        if x1-1 < 0:
            isRightBlocked = 1
            
        if y1+1 >= dis_height:
            isUpBlocked = 1
            
        if y1-1 < 0:
            isDownBlocked = 1
            
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
            
        x1 += x1_change
        y1 += y1_change
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]
 
        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True
            if x[0]+1 == snake_Head[0] and x[1] == snake_Head[1]:
                isLeftBlocked = 1
            if x[0]-1 == snake_Head[0] and x[1] == snake_Head[1]:
                isRightBlocked = 1
            if x[0] == snake_Head[0] and x[1]+1 == snake_Head[1]:
                isUpBlocked = 1
            if x[0] == snake_Head[0] and x[1]-1 == snake_Head[1]:
                isDownBlocked = 1
                
 
        our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
 
        pygame.display.update()
 
        if x1 == foodx and y1 == foody:
            foodx, foody = placeFruit()
            Length_of_snake += 1
 
        clock.tick(snake_speed)
        moves.append(''.join(map(str,[isLeftBlocked, isRightBlocked, isUpBlocked, isDownBlocked, suggestedDirection, '\n'])))
    
    training_file = open("trainingData.txt","w+")
    training_file.writelines(moves)
    training_file.close()
    pygame.quit()
    quit()
 
 
gameLoop(currentGame, moves)