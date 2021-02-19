import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
import plotly.express as px
import pandas as pd

st.header("Epsilon Greedy v0")
st.subheader("Proof of concept for reinforcement learning algorithm")
st.subheader("AKA - The Multi-Armed Bandit")
st.write('Clicking the "Compute" button below will simulate the multi-armed bandit problem.  What is the multi-armed bandit problem?')
st.write('Imagine you are in a casino and walk up to ten identical slot machines that say "Play for Free!".  Each machine has a different average payout between $0 and $10.  Naturally, you would want to play whichever machine has the highest average payout, but how do you know which one that is?')
st.write('See this article for a more thorough description: ', 'https://www.datacamp.com/community/tutorials/introduction-reinforcement-learning')

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




if st.button('Compute'):
	st.write("Let's Gamble")
	st.write("Playing the multi-armed bandit 200 times real quick.  Check out how my average payout increases each time I play because I'm learning which patterns work and get smarter each time!")
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
	fig = px.scatter(df, x="Number of times played", y="Average Reward", range_x=[0,200], range_y=[0,10])
	fig.update_traces(marker=dict(size=16,
	                              line=dict(width=2,
	                                        color='DarkSlateGrey')),
	                  selector=dict(mode='markers+lines'))
	st.plotly_chart(fig)
	st.write("Try it again. Since you start from 0 each time, notice how the first 50 turns or so are different, but with enough plays and cumulative learnings the average payout will always increase.")

	st.write('See it in motion!')
	fig = px.scatter(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,100], range_y=[0,10])
	fig.update_traces(marker=dict(size=16,
	                              line=dict(width=2,
	                                        color='DarkSlateGrey')),
	                  selector=dict(mode='markers'))
	fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

	#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
	st.plotly_chart(fig)


