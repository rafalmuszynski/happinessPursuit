# -*- coding: utf-8 -*-



"""
The following code allows to play a game between 2 trained agents.
To run it you need to:

1) create a folder named checkpoints_A and upload checkpoints for the model
controlling the paddle on the left

2) create a folder named checkpoints_B and upload checkpoints for the model
controlling the paddle on the right
"""



# libraries
import pong_L_R as pong
import tensorflow as tf
import cv2
import numpy as np



# constant
ACTIONS = 3 #up,down, stay



# create a tensorflow graph
def createGraph():
     with tf.device('/gpu:0'):
        W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 4, 32], stddev=0.02))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))

        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
        b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))

        W_fc5 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.02))
        b_fc5 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

        s = tf.placeholder("float", [None, 60, 60, 4])

        conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "SAME") + b_conv1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides = [1, 2, 2, 1], padding = "SAME") + b_conv2)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "SAME") + b_conv3)

        conv3_flat = tf.reshape(conv3, [-1, 1024])

        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
        
        fc5 = tf.matmul(fc4, W_fc5) + b_fc5

        return s, fc5



# playing a match between 2 DQNs 
def main():
    #initialize the game
    game = pong.PongGame()
    
    # get intial frame
    frame = game.getPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (60, 60)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    # create and initialize graphs
    g1 = tf.Graph()
    g2 = tf.Graph()

    with g1.as_default():
        inp_A, out_A = createGraph()
        saver_A = tf.train.Saver(tf.global_variables())
        sess_A = tf.Session(graph=g1)
        checkpoint_A = tf.train.latest_checkpoint('./checkpoints_A')
        saver_A.restore(sess_A, checkpoint_A)
        
    with g2.as_default():
        inp_B, out_B = createGraph()
        saver_B = tf.train.Saver(tf.global_variables())
        sess_B = tf.Session(graph=g2)
        checkpoint_B = tf.train.latest_checkpoint('./checkpoints_B')
        saver_B.restore(sess_B, checkpoint_B)
    
    # keep track of timesteps
    t = 0

    while(1):
        # output tensor
        out_t_A = out_A.eval(session=sess_A, feed_dict = {inp_A : [inp_t]})[0]
        out_t_B = out_B.eval(session=sess_B, feed_dict = {inp_B : [inp_t]})[0]
        # argmax function
        argmax_t_A = np.zeros([ACTIONS])
        argmax_t_B = np.zeros([ACTIONS])

        maxIndex_A = np.argmax(out_t_A)
        maxIndex_B = np.argmax(out_t_B)
        argmax_t_A[maxIndex_A] = 1
        argmax_t_B[maxIndex_B] = 1
    
        # reward tensor
        score1, score2, cumScore1, cumScore2, rewardID_player1, rewardID_player2, cumID1, cumID2, \
        rewardSE_player1, rewardSE_player2, cumSE1, cumSE2, frame = game.getNextFrame(argmax_t_A, argmax_t_B)
        
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (60, 60)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (60, 60, 1))
        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
             
        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t + 1   

        # save stats log
        if score1 == 1 or score2 == 1:
            with open('stats_test.txt', 'a') as log:
                scoreline = 'TIMESTEP ' + str(t) + ' cumScore1 ' + str(cumScore1) + ' cumScore2 ' + str(cumScore2) \
                + ' ID1 ' + str(rewardID_player1) + ' ID2 ' + str(rewardID_player2) + ' cumID1 ' + str(cumID1) \
                + ' cumID2 ' + str(cumID2) + ' SE1 ' + str(rewardSE_player1) + ' SE2 ' + str(rewardSE_player2) \
                + ' cumSE1 ' + str(cumSE1) + ' cumSE2 ' + str(cumSE2) + '\n'
                log.write(scoreline)
            
        print("TIMESTEP", t, "/ EPSILON", "0")



if __name__ == "__main__":
    main()