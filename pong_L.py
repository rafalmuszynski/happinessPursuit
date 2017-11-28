# -*- coding: utf-8 -*-



"""
Credits:
The following pong code is a modification of https://github.com/malreddysid/pong_RL
"""



# libraries
import pygame
import random
import math
import rewards as rew



# constants
# size of the window
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

# size of each paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

# distance from the edge of the window
PADDLE_BUFFER = 10

# size of the ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

# vertical speed of each paddle
PADDLE_SPEED = 5
# horizontal and vertical speed of the ball
BALL_X_SPEED = 10
BALL_Y_SPEED = 7

# RGB colors used - black background, white ball and paddles
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



# initialize the screen of a given size
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))



def drawBall(ballXPos, ballYPos):
    """ creates and draws a rectangle - ball - at a given location """
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)



def drawPaddle1(paddle1YPos):
    """ creates and draws a paddle at a given location for the 1st player """
    paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle1)



def drawPaddle2(paddle2YPos):
    """ creates and draws a paddle at a given location for the 2nd player """
    paddle2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle2)



def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballAngle):
    """ computes the new ball position given the position of both paddles, its own postion and direction, also provides the score """
    
    # update the x and y position, set scores for each player
    ballXPos = ballXPos + ballXDirection * BALL_X_SPEED *  math.cos(ballAngle)
    ballYPos = ballYPos + ballXDirection * BALL_Y_SPEED * -math.sin(ballAngle)
    score1 = 0
    score2 = 0

    # LEFT SIDE
    # checks for a collision with paddle1
    # NOTE: have to keep <= in the first condition as the values do not exactly match, due to constants
    
    if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle1YPos and
        ballYPos <= paddle1YPos + PADDLE_HEIGHT):
        
        # switch direction of the ball
        ballXDirection = 1
        
        # compute and change the angle of reflection
        diff = (paddle1YPos + PADDLE_HEIGHT / 2) - (ballYPos + BALL_HEIGHT / 2)
        
        if diff != 0:
            ballAngle = math.radians(diff/35*80) # 35 is the assumed max difference between the center of paddle and ball - serves as a normalizing constant, 80 (from 0-90 degrees range) added arbitrarly to make the angles more acute
        else:
            ballAngle = math.radians(0)
        
    # if it goes past paddle 1
    elif (ballXPos <= 0):
        
        # give a point to player 2
        score1 = 0
        score2 = 1
        
        # redraw the ball in the middle
        ballXPos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        ballYPos = WINDOW_HEIGHT / 2 - BALL_HEIGHT / 2
        
        # randomly send the ball towards one of the players at 70 degrees angle
        num = random.randint(0,1)
        
        if(num == 0):
            ballXDirection = 1
        if(num == 1):
            ballXDirection = -1

        ballAngle = math.radians(70)
                
        return [score1, score2, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballAngle]
    
    # RIGHT SIDE
    # check for a collision with paddle2       
    
    if (ballXPos + BALL_WIDTH >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and ballYPos + BALL_HEIGHT >= paddle2YPos and
        ballYPos <= paddle2YPos + PADDLE_HEIGHT):
        
        # switch direction of the ball
        ballXDirection = -1
        
        # compute and change the angle of reflection
        diff = (paddle2YPos + PADDLE_HEIGHT / 2) - (ballYPos + BALL_HEIGHT / 2)
        
        if diff != 0:
            ballAngle = math.radians(-diff/35*80) # 35 is the assumed max difference between the center of paddle and ball - serves as a normalizing constant, 80 added arbitrarly to make the angles more acute
        else:
            ballAngle = math.radians(0)

    # if it goes past paddle 2
    elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
        
        # give a point to player 1
        score1 = 1
        score2 = 0
        
        # redraw the ball in the middle
        ballXPos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        ballYPos = WINDOW_HEIGHT / 2 - BALL_HEIGHT / 2
        
        # randomly send the ball towards one of the players at 70 degrees angle
        num = random.randint(0,1)
        
        if(num == 0):
            ballXDirection = 1
        if(num == 1):
            ballXDirection = -1

        ballAngle = math.radians(70)
                
        return [score1, score2, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballAngle]
    
    # TOP AND BOTTOM
    # if it hits the top - move down
    if (ballYPos <= 0):
        ballYPos = 0
        ballAngle *= -1
        
    # if it hits the bottom - move up
    elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
        ballAngle *= -1
    
    return [score1, score2, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballAngle]



def updatePaddle1(action, paddle1YPos):
    """ updates paddle position for the agent """
    
    # if move up
    if (action[1] == 1):
        paddle1YPos = paddle1YPos - PADDLE_SPEED
    # if move down
    if (action[2] == 1):
        paddle1YPos = paddle1YPos + PADDLE_SPEED
    # don't let it move off the screen
    if (paddle1YPos < 0):
        paddle1YPos = 0
    if (paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    
    return paddle1YPos


    
# hand-coded AI for the opponent
def updatePaddle2(paddle2YPos, ballYPos):
    """ updates paddle position for the hand-coded AI """
    
    if (paddle2YPos + PADDLE_HEIGHT / 2 < ballYPos + BALL_HEIGHT / 2):
        paddle2YPos = paddle2YPos + PADDLE_SPEED
    # move up if ball is in lower half
    if (paddle2YPos + PADDLE_HEIGHT / 2 > ballYPos + BALL_HEIGHT / 2):
        paddle2YPos = paddle2YPos - PADDLE_SPEED
    # don't let it hit top
    if (paddle2YPos < 0):
        paddle2YPos = 0
    # dont let it hit bottom
    if (paddle2YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    
    return paddle2YPos

    

def drawScore(score1, score2):
    """ draws score on screen """
    
    font = pygame.font.Font(None, 32)
    scorelbl_1 = font.render(str(score1), 1, WHITE)
    scorelbl_2 = font.render(str(score2), 1, WHITE)
    screen.blit(scorelbl_1, (100 , 20))
    screen.blit(scorelbl_2, (300 , 20))


    
# game class
class PongGame:
    def __init__(self):
        pygame.font.init()

        # initialize positions of paddles
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        # initialize position of the ball
        self.ballXPos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        self.ballYPos = WINDOW_HEIGHT / 2 - BALL_HEIGHT / 2
        # randomly initialize ball direction
        self.ballAngle = math.radians(0)
        
        num = random.randint(0,1)

        if(num == 0):
            self.ballXDirection = 1
        if(num == 1):
            self.ballXDirection = -1

        # initialize the cumulative score for player 1 and 2
        self.tally1 = 0
        self.tally2 = 0
        # initialize variable for cumulative rewards for both players
        self.cumID1 = 0
        self.cumID2 = 0
        self.cumSE1 = 0
        self.cumSE2 = 0

    def getPresentFrame(self):
        # for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        # make the background black
        screen.fill(BLACK)
        # draw our paddles
        drawPaddle1(self.paddle1YPos)
        drawPaddle2(self.paddle2YPos)
        # draw our ball and the score
        drawBall(self.ballXPos, self.ballYPos)
        drawScore(self.tally1, self.tally2)  
        # copies the pixels from our surface to a 3D array, to be used by the DRL algo
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # updates the window
        pygame.display.flip()

        #return surface data
        return image_data

    def getNextFrame(self, action):
        pygame.event.pump()
        screen.fill(BLACK)
        
        # update position and draw paddle 1
        self.paddle1YPos = updatePaddle1(action, self.paddle1YPos)
        drawPaddle1(self.paddle1YPos)
        # update position and draw paddle 2
        self.paddle2YPos = updatePaddle2(self.paddle2YPos, self.ballYPos)
        drawPaddle2(self.paddle2YPos)
        # update ball position and score stats, draw ball and score
        [score1, score2, self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballAngle] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballAngle)
        drawBall(self.ballXPos, self.ballYPos)
        
        # scores and rewards
        
        # record the total score
        self.tally1 = self.tally1 + score1
        self.tally2 = self.tally2 + score2

        drawScore(self.tally1, self.tally2)
        
        # compute the rewards
        # ID
        rewardID_player1, rewardID_player2 = rew.rewardID(score1, score2)
        
        self.cumID1 = self.cumID1 + rewardID_player1
        self.cumID2 = self.cumID2 + rewardID_player2
        
        # SUPEREGO
        rewardSE_player1 = rew.rewardSE_L(score1, score2, self.tally1, self.tally2)
        rewardSE_player2 = rew.rewardSE_R(score1, score2, self.tally1, self.tally2)
        self.cumSE1 = self.cumSE1 + rewardSE_player1
        self.cumSE2 = self.cumSE2 + rewardSE_player2
        
        # get the surface data
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # update the window
        pygame.display.flip()
                
        # increase score until 11 points reached
        # return the score, rewards and surface data
        
        if self.tally1 == 11 or self.tally2 == 11:
            data = [score1, score2, self.tally1, self.tally2, rewardID_player1, rewardID_player2, self.cumID1, self.cumID2, \
            rewardSE_player1, rewardSE_player2, self.cumSE1, self.cumSE2, image_data]
            
            self.tally1 = 0
            self.tally2 = 0
            # restart the cumulative reward variables
            self.cumID1 = 0
            self.cumID2 = 0
            self.cumSE1 = 0
            self.cumSE2 = 0
      
            return data
        
        else:
            return [score1, score2, self.tally1, self.tally2, rewardID_player1, rewardID_player2, self.cumID1, self.cumID2, \
                rewardSE_player1, rewardSE_player2, self.cumSE1, self.cumSE2, image_data]