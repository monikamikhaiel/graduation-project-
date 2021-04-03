import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

# area

SIZE_W = 5
SIZE_H = 5


# In[ ]:


HM_EPISODES = 25000
mediumSNR_PENALTY = 10  # feel free to tinker with these!
lowSNR_PENALTY = 40  # feel free to tinker with these!
highSNR_REWARD = 25  # feel free to tinker with these!

epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 100  # how often to play through env visually.

start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95

ANTENNA_N = 1  # player key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0) }  # Antenna color blue
#SNR RANGES
snr_range={"highSNR_max":-50,"highSNR_min":-80,
           "mediumSNR_max":-90,"mediumSNR_min":-100,
          "deadzone":-120}


# In[ ]:


#antenna
class antenna:
    def __init__(self):
        self.x = np.random.randint(0, SIZE_W)
        self.y = np.random.randint(0, SIZE_H)
    def __str__(self):
        return f"{self.x}, {self.y}"
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:  #up
            self.move(x=0, y=1)
        elif choice == 1:  #down
            self.move(x=0, y=-1)
        elif choice == 2:  #left
            self.move(x=-1, y=0)
        elif choice == 3:    #right
            self.move(x=1, y=0)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE_W-1:
            self.x = SIZE_W-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_H-1:
            self.y = SIZE_H-1



# In[ ]:


ant = antenna()
print(ant)
ant.action(0)
print(ant)


# In[ ]:


x = SIZE_W # Building X
y = SIZE_H # Building Y
z = 4 # actions

q_table = np.zeros((x, y, z))
# q_table *= 0
print(q_table.shape)


# In[ ]:


def measureSNR (pos):
#    # Gain_pannel=7 #in db
# #     Gain_Mobile=1
#     pathloss=Gain_pannel*Gain_Mobile*

#     x=input("please enter snr for (" + pos.x + ", " + pos.y + ") ? ")
#     return x
    return np.random.random((1,(SIZE_W*SIZE_H)))


# In[ ]:


def calSNRlow(grid):
    counter =0
    for i in range(1,len(grid)):
        if int(grid[i]) < snr_range["deadzone"]:
            counter+=1
    percent=counter*100/len(grid)
    return percent


# In[ ]:


episode_rewards = []
reward=0
for episode in range(HM_EPISODES):
    ant = antenna()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}")
#         print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    state=0
    for i in range(200):
        antennaPos = (ant) # Current Antenna X, Y Co-ord.

        if np.random.random() > epsilon:
            # GET THE ACTION
#             if state <=25:
#                 state=0
#             else:
#                 state+=1

            action = np.argmax(q_table[ant.x][ant.y])
        else:
            action = np.random.randint(0, 3)
        # Take the action!
        ant.action(action)
        ###
        #calculate the SNR
        snr=measureSNR(ant)
        percent=calSNRlow(snr)
        ##
        if percent > 20:
            reward = -10
        else:
            reward = 20
          ###
        #current_q = q_table[state][action]
        newAntennaPos = (ant)
        current_q = q_table[ant.x][ant.y][action] # Current State
        max_future_q = np.max(q_table[ant.x][ant.y]) # Max State
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[ant.x][ant.y][action] = new_q


        episode_reward += reward

        print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


# In[ ]:


if snr >snr_range["highSNR_min"]:
    reward+= highSNR_REWARD
elif snr < snr_range["mediumSNR_max"] and  snr > snr_range["mediumSNR_min"]:
    reward+= mediumSNR_PENALTY
elif snr < snr_range["deadzone"]:
    reward-=lowSNR_PENALTY


# In[ ]:


print(episode_reward)

