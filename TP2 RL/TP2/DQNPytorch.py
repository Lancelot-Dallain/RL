# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 16:45:34 2026

@author: barthes
"""
#%%
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#import math
#import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from network import Network


#%% Hyperparamètres
MEM_SIZE = 4096                    # Taille de la mémoire du buffer Replay (4096)
BATCH_SIZE = 64                    # taille du batch
EXPLORATION_MIN = 0.05             # pourcentage min d'exploration
MAX_STEP=150                       # nombre max d'itérations dans une séquence (150)
TARGET_UPDATE_FREQ = 200           # Mise à jour du reseau du reseau target tous les 200 steps
FC1_DIMS=128                        # Nombre neurones premiere couche cachée
FC2_DIMS=128                        # Nombre neurones deuxième couche cachée
lr=0.0001                          # learning rate
exploration_decreasing_decay=7     # vitesse de décroissance mode exploration (7)
gamma=0.95                         # pondération cumulated reward (0.95)
#%%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
    
class ReplayBuffer:
    def __init__(self, observation_space):
        self.mem_count = 0
        
        self.states  = np.zeros((MEM_SIZE, observation_space), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, observation_space), dtype=np.float32)
        self.dones   = np.zeros(MEM_SIZE, dtype=np.float32)   # float mask

    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index]   = 1.0 - float(done)
        

        self.mem_count += 1
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
#%% Algorithme Deep Q learning
class DQN_Solver:
    def __init__(self,env,lr=0.0005):
        self.memory = ReplayBuffer(env.state_space)
        self.exploration_rate = 1.0
        self.network = Network(env.action_space,env.state_space,lr,FC1_DIMS,FC2_DIMS)
        self.action_space = env.action_space
        self.observation_space = env.state_space
        self.env=env
        
        
        self.target_network = Network(env.action_space, env.state_space, lr, FC1_DIMS, FC2_DIMS)            # Création du réseau target
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.learn_step_counter = 0


    def choose_action(self, observation):
         if random.random() < self.exploration_rate:
             return random.randrange(self.action_space)
     
         state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEVICE)
         q_values = self.network(state)
         return torch.argmax(q_values).item()

    
    def learn(self, gamma):
        if self.memory.mem_count < BATCH_SIZE:
            return 0
        
        self.learn_step_counter += 1
    
        states, actions, rewards, states_, dones = self.memory.sample()
    
        states  = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones   = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    
        batch_indices = np.arange(BATCH_SIZE)
    
        q_values = self.network(states)
        q_pred = q_values[batch_indices, actions]
    
        with torch.no_grad():
            q_next = self.target_network(states_)
            q_next_max = torch.max(q_next, dim=1)[0]
            q_target = rewards + gamma * q_next_max * dones
    
        loss = self.network.loss(q_pred, q_target)
    
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        
        #  Mise à jour périodique du target network
        if self.learn_step_counter % TARGET_UPDATE_FREQ == 0:
            
            self.target_network.load_state_dict(self.network.state_dict())
    
        return loss.item()


    def returning_epsilon(self):
        return self.exploration_rate


    def train(self,Nbr_episodes=100, filesave= None,display=True):
        best_reward = 0
        average_reward = 0
        epsilon0 = 1.0
        episode_number = []
        rewards_per_episode = []
        Loss_per_episode=[]
       
        for episode in range(1, Nbr_episodes+1):
            state = self.env.reset(x0=None,y0=None)
            state = np.reshape(state, [self.observation_space])
            
            score = 0
            step=0
            Loss=0
            while step <= MAX_STEP:      # on itère au maximum MAX-STEP par épisode
                step +=1
                action = self.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                state_ = np.reshape(state_, [self.observation_space])
                self.memory.add(state, action, reward, state_, done)
                Loss= Loss+ self.learn(gamma)
                state = state_
                score += reward
               
                if done ==1:    # Si fin du jeu on arrete l'épisode en cours
                    if score > best_reward:
                        best_reward = score
                    average_reward += score 
                    print("Done Episode {} Average Reward {:.1f} Best Reward {:.1f} Last Reward {:.1f} Epsilon {:.2f}".format(episode, average_reward/episode, best_reward, score, self.returning_epsilon()))
                    break

                   
            episode_number.append(episode)
            rewards_per_episode.append(score)
            Loss_per_episode.append(Loss)
            if step >MAX_STEP:
                print('Max step atteint Loss = ',Loss)
            else:
                print('Bravo Loss = ',Loss) 
                
            self.exploration_rate = max(EXPLORATION_MIN, epsilon0*np.exp(-exploration_decreasing_decay*episode/Nbr_episodes))
           
            print("Episode {}/{} Cum. Reward {:.1f} Epsilon {:.2f}".format(episode, Nbr_episodes,score,self.returning_epsilon()))
        
        if filesave is not None:
            torch.save(self.network.state_dict(), filesave)
            print('Saved in ',filesave )
        
        if display:
            plt.figure()
            plt.plot(episode_number,  rewards_per_episode)
            plt.xlabel('Episodes')
            plt.ylabel('Cumulated reward')
            plt.show()
