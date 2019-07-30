import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500         ##number of steps in game
score_requirement = 60
initial_games = 10000    ##total number of games

def model_data_prep():

        training_data = []
        accepted_scores = []

        for game_index in range(initial_games):
                
                score = 0
                game_memory = []
                previous_observation = []

                ## playing game for 500 steps means compleating the game

                for step_index in range(goal_steps):

                        action  = random.randrange(0,2)
                        observation, reward, done, info = env.step(action)

                        ##try puting obeervarion in previous_obser block above if statement

                        if len(previous_observation)>0:
                                game_memory.append([previous_observation,action])
                        
                        previous_observation = observation
                        score +=reward
                        if done:
                                break
                
                if score >= score_requirement:
                        
                        accepted_scores.append(score)
                        
                        for data in game_memory:
                                if data[1] == 0:
                                        #print(data[1])
                                        output = [1,0]
                                elif data[1] == 1:
                                        #print(data[1])
                                        output =  [0,1]
                                training_data.append([data[0], output])
                               # print(training_data)
                env.reset()
        
        #print(accepted_scores)
        return training_data


training_data = model_data_prep()


def build_model(input_siz, output_siz):
        
        model = Sequential()  ##https://machinelearningmastery.com/keras-functional-api-deep-learning/
        model.add(Dense(128, input_dim = input_siz, activation = 'relu'))
        model.add(Dense(64,activation = 'relu'))
        model.add(Dense(output_siz, activation = 'softmax')) #try softmax, it works
        model.compile(loss = 'mean_squared_error',optimizer = Adam()) #https://keras.io/optimizers/ 
        return model

##needs time!!!
def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_siz=len(x[0]), output_siz=len(y[0]))
    
    model.fit(x, y, epochs=10)
    return model

trained_model = train_model(training_data)


#final playing of game bot

scores = []
choices = []

for each_game in range(100):
        score = 0
        prev_ober = []
        game_memory = []
        for step_index in range(goal_steps):
               
                env.render()
                #print(np.shape(prev_ober))
                if len(prev_ober) == 0:
                        action = random.randrange(0,2)
                else:
                        action = np.argmax(trained_model.predict(prev_ober.reshape(-1,len(prev_ober)))[0])

                choices.append(action)
                new_observation, reward, done, info = env.step(action)
                prev_ober = new_observation
                game_memory.append([new_observation, action])
                score += reward
                if done:
                        break

        env.reset()
        scores.append(score)


#print(scores)
#print('Average Score: ', sum(scores)/len(scores))
#print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices))) ## ???

#the end :)