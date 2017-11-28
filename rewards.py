# -*- coding: utf-8 -*-



"""
The following code specifies the reward signals provided to the agents
in the following order:
1) reward corresponding to ID values
2) reward corresponding to SE values defined for agent controlling the
paddle on the left or right
"""



def rewardID(score1, score2):
    """ computes ID reward, +1 if player scores a point, -1 otherwise """
    
    reward_player1 = 0
    reward_player2 = 0
    
    if score1 == 1:
        reward_player1 = 1
        reward_player2 = -1
    elif score2 == 1:
        reward_player1 = -1
        reward_player2 = 1
    
    return reward_player1, reward_player2



def rewardSE_L(score1, score2, cum_score1, cum_score2):
    """ computes SE reward, depending on whether the agent is 'catching up'
    or 'getting ahead' """
    
    reward1 = 0
    
    if score1 == 1 and score2 == 0:				
        diff = cum_score1 - cum_score2			
        if diff <= 1:			
            reward1 = 1		
        elif diff > 1:			
            reward1 = 1.0/diff
    
    if score1 == 0 and score2 == 1:				
        diff = cum_score1 - cum_score2			
        if diff == 0:			
            reward1 = -0.05		
        if diff < 0:	
            reward1 = diff/11.0		
        if diff > 0:			
            reward1 = diff/(-11.0)		
    
    return reward1
    

    
def rewardSE_R(score1, score2, cum_score1, cum_score2):
    """ computes SE reward, depending on whether the agent is 'catching up'
    or 'getting ahead' """
    
    reward2 = 0
    
    if score1 == 0 and score2 == 1:				
        diff = cum_score2 - cum_score1			
        if diff <= 1:			
            reward2 = 1		
        elif diff > 1:			
            reward2 = 1.0/diff
    
    if score1 == 1 and score2 == 0:				
        diff = cum_score2 - cum_score1			
        if diff == 0:			
            reward2 = -0.05		
        elif diff < 0:	
            reward2 = diff/11.0		
        elif diff > 0:			
            reward2 = diff/(-11.0)		
    
    return reward2
