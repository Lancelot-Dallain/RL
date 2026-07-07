# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:50:12 2023
Modifié le 01/2026
@author: barthes
"""
import numpy as np
import matplotlib.pyplot as plt
import random
def Qlearning(env,action_size, state_size,total_episode=1500, lr=0.8,gamma=0.9,exploration_decreasing_decay=5,max_steps = 150 ,min_exploration_proba = 0.01,display=True, board_name = ""):
    q_table = np.zeros((state_size, action_size))
    Nstates=np.zeros(state_size)    # table pour compter le nombre de fois qu'un état est visité
    # Hyper params:
    #max_steps = 150            # nombre max de steps par épisode
    #min_exploration_proba = 0.01     # probabilité min du mode exploration
    #exploration_decreasing_decay=5   # vitesse de décroissance de epsilon
    # gamma : ponderation cumulated reward
    
    epsilon0 = 1.0             # initialement mode exploration = 100% 
    rewards_per_episode = []   # liste pour stocker les cumulated rewards
    epsilon=epsilon0
    
    for episode in range(total_episode):                      # boucle sur les episodes
        # Reset Environment:
        current_state = env.reset(x0=None,y0=None)    # position aléatoire et salades mangées aléatoires
        total_episode_reward = 0    #sum the rewards that the agent gets from the environment
        
        print('Episode {:d}/{:d} Eps = {:.2f}'.format(episode,total_episode,epsilon))
        for step in range(max_steps):      # boucle sur les pas de temps
            
            exploitation_tradeoff = random.uniform(0, 1)
            # If this number > greater than epsilon → exploitation (taking the biggest q value for the current state):
            if exploitation_tradeoff > epsilon:
                action = np.argmax(q_table[current_state, :])
            # Else, doing random choice:
            else:
                action = env.sample()
                
            # Take the action (a) and observe the outcome state and the reward 
            next_state, reward, done, info = env.step(action)
            Nstates[next_state] +=1    # histogramme des etats
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s’,a’) - Q(s,a)]
            q_table[current_state,action] = q_table[current_state, action] + lr * (
    reward + gamma * np.max(q_table[next_state, :]) - q_table[current_state, action]
)
            
            total_episode_reward = total_episode_reward + reward   # Calcul reward cumulé par episode
            
            # If done True, finish the episode:  
            if done == True:
                break
            # Our new state:
            current_state = next_state  
        # Reduce epsilon (because we need less and less exploration):
        #We update the exploration proba using exponential decay formula 
        epsilon = max(min_exploration_proba, epsilon0*np.exp(-exploration_decreasing_decay*episode/total_episode))
        # Sauvegarde total_episode_reward
        rewards_per_episode.append(total_episode_reward)
    
    if display:
        plt.figure()
        plt.plot(rewards_per_episode)
        plt.ylabel('Cumulated reward / episode')
        plt.xlabel('Episodes')
        plt.title('lr : {}  gamma : {}  decay : {}'.format(lr,gamma,exploration_decreasing_decay))
        plt.figure()
        plt.plot(Nstates)
        plt.xlabel('States')
        plt.ylabel('Hist.')
        # on veut voir le graph
        plt.show()

    

    actions_labels = ['Down', 'Left', 'Up', 'Right']
    best_actions = np.argmax(q_table, axis=1)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(q_table, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Q-value')

    plt.xlabel('Actions')
    plt.ylabel('States')

    plt.title(f"Q-table heatmap (★ = best action), Board: {board_name} | lr={lr}, gamma={gamma}, decay={exploration_decreasing_decay}")

    plt.xticks(range(q_table.shape[1]), actions_labels)
    plt.yticks(range(q_table.shape[0]))

    for s in range(q_table.shape[0]):
        for a in range(q_table.shape[1]):
            label = f"{q_table[s, a]:.2f}"
            if a == best_actions[s]:
                label += " ★"
            plt.text(
                a, s, label,
                ha="center", va="center",
                fontsize=15,
                color="white" if q_table[s, a] > np.max(q_table) * 0.5 else "black"
            )




    plt.tight_layout()
    plt.show()


    return q_table
    