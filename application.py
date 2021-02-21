import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
import plotly.express as px
import pandas as pd


report = st.sidebar.selectbox("Epsilon Greedy Model",["Epsilon Greedy v0", "Epsilon Greedy v0.1", "Epsilon Greedy v0.2", "Epsilon Greedy v0.3", "Epsilon Greedy v0.4", "Epsilon Greedy v0.5", "Epsilon Greedy v0.6"])

if report == "Epsilon Greedy v0.6":
	st.header("Epsilon Greedy v0.6")
	st.subheader("7th Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - Using real arriv check-in data and pretending we actually did better sometimes")
	st.write('Clicking the "Compute" button below will initialize a curve assuming a benchmark conversion rate. It will then introduce small variations to the initial curve and accept those variations when they perform better.')
	st.write('For Epsilon Greedy v0.6, we are now going to introduce an actual curve and simulate how it might evolve over time and use arriv patient data to determine if a check-in would have occurred at each step.  We will introduce small fluctuations to a random bar 10% of the time and observe whether those changes are accepted or rejected and how that curve evolves over time.  In addition, we will also choose to randomly convert someone that may not have converted before.  This should allow us to simulate what it would be like if this model got smarter and was able to convert more patients.')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=1

	# this is the conversion rate of the initial curve
	curves = [0.3]

	# initialize the bimodal curve
	shape = [0.41111229, 0.95, 0.41144775, 0.057131, 0.41144775, 0.95, 0.41111229]
	slot = [1,2,3,4,5,6,7]

	st.subheader('Initial Time Curve Shape')
	# plot this initial shape
	fig = px.bar(shape, orientation='h')
	st.plotly_chart(fig)

	#generate the distribution of curves and associated probabilities.
	#mu, sigma = 0.318, 0.035

	#curves = np.random.normal(mu, sigma, n)

	# do some quick sanity checks and plots
	#verify the mean
	#mean = np.mean(curves)
	#st.write('mean of conversion rates = ', mean)

	#verify the variance
	#std = np.std(curves, ddof=1)
	#st.write('standard deviation of conversion rates = ', std)

	# plot histogram of conversion rates in this model
	#count, bins, ignored = plt.hist(curves, 30, density=True)
	#fig, ax = plt.subplots() 
	#ax = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #           np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #     linewidth=2, color='r')
	#st.pyplot(fig)


	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = curves

	# Set the probability of exploration, epsilon
	eps = 0.1

	# read in check-in data for determining reward with each attempt
	checkins = pd.read_csv('simple_checkin_data.csv', names=["patient_id", "conversion"], header=1)

	#st.write(checkins)

	# probability that we convert a patient that may not have converted before
	prob = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(patient):
		patient = patient+1
		reward = np.mean(checkins['conversion'][:patient])
		#st.write(patient,reward)
		
		# Let's now say that 10% of the time (prob) the patient actually DOES check in.
		if random.random() < prob:
			reward = 1

		return reward


	# initialize memory array; has 1 row defaulted to random action index, also save the shape state
	av = np.array([np.random.randint(0,(n+1)), 0, shape]).reshape(1,3) #av = action-value

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
		st.write("Simulating 1000 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number of patients simulated", "Average Conversion Rate", "animation", "Shape"]
		df = pd.DataFrame(columns = column_names)
		for i in range(1000):
			if (i % 200 ==0):
				st.write('Simulated ', i, 'patients so far')
			if random.random() > eps: #greedy exploitation action
				choice = bestArm(av)
				thisAV = np.array([[choice, reward(i), shape]])
				av = np.concatenate((av, thisAV), axis=0)
			else: #exploration action
			# This is the key step in which we vary the available conversion rate by a fixed amount
				choice = np.where(arms == np.random.choice(arms))[0][0]
				arms[choice] = arms[choice]+0.05*(random.random() - 0.5)
				# choose random bar to vary
				bar_choice = random.choice(list(enumerate(shape)))[0]
				# add or subtract a random amount from the chosen bar value
				shape[bar_choice] = shape[bar_choice] + 0.05*(random.random() - 0.5)
				#st.write(shape[bar_choice])
				thisAV = np.array([[choice, reward(i), shape]]) #choice, reward, shape
				av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
				
			#calculate the mean reward
			runningMean = np.mean(av[:,1])
			#st.write(i, runningMean)
			#st.write(df)
			dftmp = pd.DataFrame([{"Number patients simulated": i, "Average Conversion Rate": runningMean, "animation": i, "Shape": shape}])
			#st.write(dftmp)
			df = df.append(dftmp)
		fig = px.scatter(df, x="Number patients simulated", y="Average Conversion Rate", range_x=[0,1000], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		st.write('Final "Evolved" curve shape')
		fig = px.bar(shape, orientation='h')
		#fig = px.bar(df, y=shape, orientation='h', animation_frame='animation')
		#st.write(df.Shape[0])
		#st.write(df.Shape[199])
		#fig = px.bar(df, x=df.Shape[0], y=list(shape)[0], orientation='h')
		#st.write(df)
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		st.plotly_chart(fig)

if report == "Epsilon Greedy v0.5":
	st.header("Epsilon Greedy v0.5")
	st.subheader("6th Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - Using real arriv check-in data")
	st.write('Clicking the "Compute" button below will initialize a curve assuming a benchmark conversion rate. It will then introduce small variations to the initial curve and accept those variations when they perform better.')
	st.write('For Epsilon Greedy v0.5, we are now going to introduce an actual curve and simulate how it might evolve over time and use arriv patient data to determine if a check-in would have occurred at each step.  We will introduce small fluctuations to a random bar 10% of the time and observe whether those changes are accepted or rejected and how that curve evolves over time.')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=1

	# this is the conversion rate of the initial curve
	curves = [0.3]

	# initialize the bimodal curve
	shape = [0.41111229, 0.95, 0.41144775, 0.057131, 0.41144775, 0.95, 0.41111229]
	slot = [1,2,3,4,5,6,7]

	st.subheader('Initial Time Curve Shape')
	# plot this initial shape
	fig = px.bar(shape, orientation='h')
	st.plotly_chart(fig)

	#generate the distribution of curves and associated probabilities.
	#mu, sigma = 0.318, 0.035

	#curves = np.random.normal(mu, sigma, n)

	# do some quick sanity checks and plots
	#verify the mean
	#mean = np.mean(curves)
	#st.write('mean of conversion rates = ', mean)

	#verify the variance
	#std = np.std(curves, ddof=1)
	#st.write('standard deviation of conversion rates = ', std)

	# plot histogram of conversion rates in this model
	#count, bins, ignored = plt.hist(curves, 30, density=True)
	#fig, ax = plt.subplots() 
	#ax = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #           np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #     linewidth=2, color='r')
	#st.pyplot(fig)


	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = curves

	# Set the probability of exploration, epsilon
	eps = 0.1

	# read in check-in data for determining reward with each attempt
	checkins = pd.read_csv('simple_checkin_data.csv', names=["patient_id", "conversion"], header=1)

	#st.write(checkins)

	prob = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(patient):
		patient = patient+1
		reward = np.mean(checkins['conversion'][:patient])
		#st.write(patient,reward)

		return reward


	# initialize memory array; has 1 row defaulted to random action index, also save the shape state
	av = np.array([np.random.randint(0,(n+1)), 0, shape]).reshape(1,3) #av = action-value

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
		st.write("Simulating 1000 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number of patients simulated", "Average Conversion Rate", "animation", "Shape"]
		df = pd.DataFrame(columns = column_names)
		for i in range(1000):
			if (i % 200 ==0):
				st.write('Simulated ', i, 'patients so far')
			if random.random() > eps: #greedy exploitation action
				choice = bestArm(av)
				thisAV = np.array([[choice, reward(i), shape]])
				av = np.concatenate((av, thisAV), axis=0)
			else: #exploration action
			# This is the key step in which we vary the available conversion rate by a fixed amount
				choice = np.where(arms == np.random.choice(arms))[0][0]
				arms[choice] = arms[choice]+0.05*(random.random() - 0.5)
				# choose random bar to vary
				bar_choice = random.choice(list(enumerate(shape)))[0]
				# add or subtract a random amount from the chosen bar value
				shape[bar_choice] = shape[bar_choice] + 0.05*(random.random() - 0.5)
				#st.write(shape[bar_choice])
				thisAV = np.array([[choice, reward(i), shape]]) #choice, reward, shape
				av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
				
			#calculate the mean reward
			runningMean = np.mean(av[:,1])
			#st.write(i, runningMean)
			#st.write(df)
			dftmp = pd.DataFrame([{"Number patients simulated": i, "Average Conversion Rate": runningMean, "animation": i, "Shape": shape}])
			#st.write(dftmp)
			df = df.append(dftmp)
		fig = px.scatter(df, x="Number patients simulated", y="Average Conversion Rate", range_x=[0,1000], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		st.write('Final "Evolved" curve shape')
		fig = px.bar(shape, orientation='h')
		#fig = px.bar(df, y=shape, orientation='h', animation_frame='animation')
		#st.write(df.Shape[0])
		#st.write(df.Shape[199])
		#fig = px.bar(df, x=df.Shape[0], y=list(shape)[0], orientation='h')
		#st.write(df)
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		st.plotly_chart(fig)


if report == "Epsilon Greedy v0.4":
	st.header("Epsilon Greedy v0.4")
	st.subheader("5th Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - The one with the bars")
	st.write('Clicking the "Compute" button below will initialize a curve assuming a benchmark conversion rate. It will then introduce small variations to the initial curve and accept those variations when they perform better.')
	st.write('For Epsilon Greedy v0.4, we are now going to introduce an actual curve and simulate how it might evolve over time.  We will introduce small fluctuations to a random bar 10% of the time and observe whether those changes are accepted or rejected and how that curve evolves over time.')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=1

	# this is the conversion rate of the initial curve
	curves = [0.318]

	# initialize the bimodal curve
	shape = [0.41111229, 0.95, 0.41144775, 0.057131, 0.41144775, 0.95, 0.41111229]
	slot = [1,2,3,4,5,6,7]

	st.subheader('Initial Time Curve Shape')
	# plot this initial shape
	fig = px.bar(shape, orientation='h')
	st.plotly_chart(fig)

	#generate the distribution of curves and associated probabilities.
	#mu, sigma = 0.318, 0.035

	#curves = np.random.normal(mu, sigma, n)

	# do some quick sanity checks and plots
	#verify the mean
	#mean = np.mean(curves)
	#st.write('mean of conversion rates = ', mean)

	#verify the variance
	#std = np.std(curves, ddof=1)
	#st.write('standard deviation of conversion rates = ', std)

	# plot histogram of conversion rates in this model
	#count, bins, ignored = plt.hist(curves, 30, density=True)
	#fig, ax = plt.subplots() 
	#ax = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #           np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #     linewidth=2, color='r')
	#st.pyplot(fig)


	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = curves

	# Set the probability of exploration, epsilon
	eps = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(prob):
		reward = 0

		for i in range(1):
			if random.random() < prob:
				reward += 1

		return reward


	# initialize memory array; has 1 row defaulted to random action index, also save the shape state
	av = np.array([np.random.randint(0,(n+1)), 0, shape]).reshape(1,3) #av = action-value

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
		st.write("Simulating 1000 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number of patients simulated", "Average Conversion Rate", "animation", "Shape"]
		df = pd.DataFrame(columns = column_names)
		new_shape = shape.copy()
		for i in range(601):
			if (i % 200 ==0):
				st.write('Simulated ', i, 'patients so far')
			if random.random() > eps: #greedy exploitation action
				choice = bestArm(av)
				#new_shape = shape.copy()
				thisAV = np.array([[choice, reward(arms[choice]), new_shape]])
				av = np.concatenate((av, thisAV), axis=0)
			else: #exploration action
			# This is the key step in which we vary the available conversion rate by a fixed amount
				choice = np.where(arms == np.random.choice(arms))[0][0]
				arms[choice] = arms[choice]+0.05*(random.random() - 0.5)
				# choose random bar to vary
				bar_choice = random.choice(list(enumerate(shape)))[0]
				# add or subtract a random amount from the chosen bar value
				new_shape = new_shape.copy()
				new_shape[bar_choice] = new_shape[bar_choice] + 0.05*(random.random() - 0.5)
				#st.write('Shape updated')
				#st.write(shape[bar_choice])
				thisAV = np.array([[choice, reward(arms[choice]), new_shape]]) #choice, reward, shape
				av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
				
			#calculate the mean reward
			runningMean = np.mean(av[:,1])
			#st.write(i, runningMean)
			#st.write(df)
			dftmp = pd.DataFrame([{"Number of patients simulated": i, "Average Conversion Rate": runningMean, "animation": i, "Shape": new_shape}])
			#st.write(dftmp)
			#st.write(av)
			#df = df.append(dftmp, ignore_index=True)
			df = pd.concat([df, dftmp])
			#st.write(df)
		fig = px.scatter(df, x="Number of patients simulated", y="Average Conversion Rate", range_x=[0,1000], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		st.write('Final "Evolved" curve shape')
		fig = px.bar(new_shape, orientation='h')
		fig.update_layout(transition_duration=1)
		#fig = px.bar(df, y=shape, orientation='h', animation_frame='animation')
		#st.write(df.Shape[0])
		#st.write(df.Shape[199])
		#fig = px.bar(df, x=df.Shape[0], y=list(shape)[0], orientation='h')
		#st.write(df)
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		st.plotly_chart(fig)
		#st.write(df)

		#split shape values into their own columns
		dfplot = pd.DataFrame(df["Shape"].to_list(), columns=['slot_s1','slot_s2','slot_s3','slot_s4','slot_s5','slot_s6','slot_s7'])
		dfplot = dfplot.reset_index()
		#st.write(dfplot)


		# get data in long format instead of wide format
		dfplot = pd.wide_to_long(dfplot, stubnames='slot',
                          i=["index"], j='slots',
                          sep='_', suffix='\w+').reset_index()

		#st.write(dfplot)


		fig = px.bar(dfplot, x="slot", y='slots', animation_frame="index")
		st.plotly_chart(fig)

if report == "Epsilon Greedy v0.3":
	st.header("Epsilon Greedy v0.3")
	st.subheader("4th Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - The small variations on a well performing curve that dynamically updates the definition of the well performing curve")
	st.write('Clicking the "Compute" button below will initialize a curve assuming a benchmark conversion rate. It will then update the winning conversion rate after every check in for 1000 patient check-in attempts.')
	st.write('For Epsilon Greedy v0.3, we aren\'t actually going to specify what the well performing curve is.  We will just start with a conversion rate of 30% and assume that randomly changes about 10% of the time in a good or bad direction.  Over time we expect this to continue to increase - we will see...')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=1

	curves = [0.318]

	#generate the distribution of curves and associated probabilities.
	#mu, sigma = 0.318, 0.035

	#curves = np.random.normal(mu, sigma, n)

	# do some quick sanity checks and plots
	#verify the mean
	#mean = np.mean(curves)
	#st.write('mean of conversion rates = ', mean)

	#verify the variance
	#std = np.std(curves, ddof=1)
	#st.write('standard deviation of conversion rates = ', std)

	# plot histogram of conversion rates in this model
	#count, bins, ignored = plt.hist(curves, 30, density=True)
	#fig, ax = plt.subplots() 
	#ax = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #           np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #     linewidth=2, color='r')
	#st.pyplot(fig)


	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = curves

	# Set the probability of exploration, epsilon
	eps = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(prob):
		reward = 0

		for i in range(1):
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
		st.write("Simulating 1000 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number of patients simulated", "Average Conversion Rate"]
		df = pd.DataFrame(columns = column_names)
		for i in range(1000):
			if random.random() > eps: #greedy exploitation action
				choice = bestArm(av)
				thisAV = np.array([[choice, reward(arms[choice])]])
				av = np.concatenate((av, thisAV), axis=0)
			else: #exploration action
			# This is the key step in which we vary the available conversion rate by a fixed amount
				choice = np.where(arms == np.random.choice(arms))[0][0]
				arms[choice] = arms[choice]+0.05*(random.random() - 0.5)
				#st.write(arms[choice])
				thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward
				av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
				
			#calculate the mean reward
			runningMean = np.mean(av[:,1])
			#st.write(i, runningMean)
			#st.write(df)
			dftmp = pd.DataFrame([{"Number patients simulated": i, "Average Conversion Rate": runningMean, "animation": i}])
			#st.write(dftmp)
			df = df.append(dftmp)
		fig = px.scatter(df, x="Number patients simulated", y="Average Conversion Rate", range_x=[0,1000], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		#fig = px.scatter(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,200], range_y=[0,1])
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		#st.plotly_chart(fig)

if report == "Epsilon Greedy v0.2":
	st.header("Epsilon Greedy v0.2")
	st.subheader("3rd Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - The small variations on a well performing curve experiment")
	st.write('Clicking the "Compute" button below will simulate a well performing curve and 100 variations of it for 1000 patient check-in attempts.')
	st.write('For Epsilon Greedy v0.2, we aren\'t actually going to specify what the well performing curve is.  We will just assume that we are likely to guess a large amount of slightly better or worse variations and a small amount of really good or bad variations according to a normal distribution around a mean of the base conversion rate.  See the curve below to see the distribution of available conversion rates in this model.')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=100

	#generate the distribution of curves and associated probabilities.
	mu, sigma = 0.318, 0.035

	curves = np.random.normal(mu, sigma, n)

	# do some quick sanity checks and plots
	#verify the mean
	mean = np.mean(curves)
	st.write('mean of conversion rates = ', mean)

	#verify the variance
	std = np.std(curves, ddof=1)
	st.write('standard deviation of conversion rates = ', std)

	# plot histogram of conversion rates in this model
	count, bins, ignored = plt.hist(curves, 30, density=True)
	fig, ax = plt.subplots() 
	ax = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
	st.pyplot(fig)


	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = curves

	# Set the probability of exploration, epsilon
	eps = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(prob):
		reward = 0

		for i in range(1):
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
		st.write("Simulating 1000 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number of patients simulated", "Average Conversion Rate"]
		df = pd.DataFrame(columns = column_names)
		for i in range(1000):
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
			dftmp = pd.DataFrame([{"Number patients simulated": i, "Average Conversion Rate": runningMean, "animation": i}])
			#st.write(dftmp)
			df = df.append(dftmp)
		fig = px.scatter(df, x="Number patients simulated", y="Average Conversion Rate", range_x=[0,1000], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		#fig = px.scatter(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,200], range_y=[0,1])
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		#st.plotly_chart(fig)

if report == "Epsilon Greedy v0.1":
	st.header("Epsilon Greedy v0.1")
	st.subheader("2nd Proof of concept for reinforcement learning algorithm")
	st.subheader("AKA - The Multi-Curve Check-In Experiment")
	st.write('Clicking the "Compute" button below will simulate the multi-curve check-in experiment for 500 patient check-in attempts.  What is the multi-curve check-in experiment?')
	st.write('For Epsilon Greedy v0.1, we can think of the Multi-Curve Check-In Experiment as one in which we have predetermined conversion rates for each of 6 available time curves.  To begin with, we show the curves to users randomly.  Over time we record the outcome of each check-in attempt - this is determined by the current conversion rate for each curve, typically between 22% - 32%.  Over time we would expect the algorithm to find the best performing curve and to optimize to that value.')

	# initialize a random seed
	np.random.seed(5)

	# Set number of arms for the bandit problem
	n=6

	# Equate arms with known curve conversion rates from this google slide: https://docs.google.com/presentation/d/1yIwNGm6Q8MbohnkOEP_ww6gRrgSytVFTxdZQ2izAjcU/edit#slide=id.gbd766acf5e_1_0
	arms = [0.281, 0.265, 0.223, 0.287, 0.318, 0.266]

	# Set the probability of exploration, epsilon
	eps = 0.1


	# for every arm, loop through ten iterations and generate
	# a random number each time.  If this random number is less
	# than the probability of that arm, we add 1 to the reward.
	# after all 10 iterations we'll have a value between 1 and 10.
	def reward(prob):
		reward = 0

		for i in range(1):
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
		st.write("Simulating 500 patients using arriv real quick.")
		# Here is the main loop.  Let's play it 500 times and display a matplotlib scatter plot 
		# of the mean reward against the number of times the game is played.
		plt.xlabel("Number of patients simulated")
		plt.ylabel("Average Conversion Rate")
		column_names = ["Number patients simulated", "Average Conversion Rate"]
		df = pd.DataFrame(columns = column_names)
		for i in range(500):
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
			dftmp = pd.DataFrame([{"Number patients simulated": i, "Average Conversion Rate": runningMean, "animation": i}])
			#st.write(dftmp)
			df = df.append(dftmp)
		fig = px.scatter(df, x="Number patients simulated", y="Average Conversion Rate", range_x=[0,500], range_y=[0,1])
		fig.update_traces(marker=dict(size=16,
		                              line=dict(width=2,
		                                        color='DarkSlateGrey')),
		                  selector=dict(mode='markers+lines'))
		st.plotly_chart(fig)
		st.write("Try it again. Since you start with no a-priori knowledge each time, notice how the average conversion rate varies highly in the beginning and then settles into a steady state after enough learnings.")
		#st.write('See it in motion!')
		#fig = px.scatter(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,200], range_y=[0,1])
		#fig.update_traces(marker=dict(size=16,
		#                              line=dict(width=2,
		#                                        color='DarkSlateGrey')),
		#                  selector=dict(mode='markers'))
		#fig.update_traces(line=dict(dash="dot", width=2, color='DarkSlateGrey'), selector=dict(type='scatter', mode='lines'))

		#fig = px.line(df, x="Number of times played", y="Average Reward", animation_frame='animation', range_x=[0,500], range_y=[0,10])
		#st.plotly_chart(fig)

if report == "Epsilon Greedy v0":
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


