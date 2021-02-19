import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
import plotly.express as px
import pandas as pd

st.write("Hey There")

# initialize a random seed
np.random.seed(5)

# Set number of arms for the bandit problem
n=10

arms = np.random.rand(n)

# Set the probability of exploration, epsilon
eps = 0.1


# for every arm, loop through ten iterations and generate
# a random number each time.  If this random number is less
# than the probability of that arm, we add 1 to the reward.
# after all 10 iterations we'll have a value between 1 and 10.
def reward(prob):
	reward = 0

	for i in range(10):
		if random.random() < prob:
			reward += 1

	return reward


# initialize memory array; has 1 row defaulted to random action index
av = np.array([np.random.randint(0,(n+1)), 0]).reshape(1,2) #av = action-value

# greedy method to select best arm based on memory array
# This function accepts a memory array that stores the history of all actions and their rewards. 
# It is a 2 x k matrix where each row is an index reference to your arms array (1st element), and the reward received (2nd element). 
# For example, if a row in your memory array is [2, 8], it means that action 2 was taken (the 3rd element in our arms array) 
# and you received a reward of 8 for taking that action.
def bestArm(a):
    bestArm = 0 #default to 0
    bestMean = 0
    for u in a:
        avg = np.mean(a[np.where(a[:,0] == u[0])][:, 1]) #calculate mean reward for each action
        if bestMean < avg:
            bestMean = avg
            bestArm = u[0]
    return bestArm


# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
# of the mean reward against the number of times the game is played.
plt.xlabel("Number of times played")
plt.ylabel("Average Reward")
column_names = ["Number of times played", "Average Reward"]
df = pd.DataFrame(columns = column_names)
for i in range(200):
    if random.random() > eps: #greedy exploitation action
        choice = bestArm(av)
        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0)
    else: #exploration action
        choice = np.where(arms == np.random.choice(arms))[0][0]
        thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward
        av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
    #calculate the mean reward
    runningMean = np.mean(av[:,1])
    #st.write(i, runningMean)
    #st.write(df)
    dftmp = pd.DataFrame([{"Number of times played": i, "Average Reward": runningMean, "animation": i}])
    #st.write(dftmp)
    df = df.append(dftmp)
    #st.write(df)
    #fig, ax1 = plt.subplots() #solved by add this line
    #if i == 0:
    #    ax0 = plt.scatter(i, runningMean, ax=ax1)
    #else:
    #    ax1 = plt.scatter(i, runningMean, ax=ax1)
    
    #st.pyplot(fig)
    #if i ==1 :
    #	fig1 = px.scatter(df, x="Number of times played", y="Average Reward", range_x=[0,500], range_y=[0,10])
    #else:
    #	fig2 = px.scatter(df, x="Number of times played", y="Average Reward", range_x=[0,500], range_y=[0,10])
    #fig3 = go.Figure(data=fig1.data + fig2.data)
    #st.plotly_chart(fig3)
st.button('Compute')
fig = px.scatter(df, x="Number of times played", y="Average Reward", range_x=[0,200], range_y=[0,10])
st.plotly_chart(fig)

st.write('See it in motion!')
fig = px.scatter(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,200], range_y=[0,10])
#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
st.plotly_chart(fig)
    
    
