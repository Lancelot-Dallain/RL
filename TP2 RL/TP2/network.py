# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 11:17:37 2026

@author: barthes
Modele neuronal à 2 couches cachées de type full connected
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
class Network(torch.nn.Module):
    '''
    Définition du réseau de neurones
    Ici un simple MLP à 2 couches cachées
    et l'environement d'apprentissage
    '''
    def __init__(self,action_space,observation_space,lr,FC1_DIMS,FC2_DIMS):
        super().__init__()
       
        self.input_shape = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space, FC1_DIMS)   # input full connected 
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)            # hidden layer
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)   # output layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)   # optimiser ADAM
        self.loss = nn.MSELoss()        # fonction de cout MSE
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def play1step(self,state):        # Calcule la sortie et choisi l'action ayant la Q value la plus forte
      
      act_values = self.forward(torch.tensor(state))  
      action= np.argmax(act_values.detach().numpy())
      return action