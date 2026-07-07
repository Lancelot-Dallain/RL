# -*- coding: utf-8 -*-
#--------------------------------
# CrÃ©ation d'un environnement de jeu de la tortue et des salades !
# Apprentissage à l'aide d'un réseau de neurones et de l'algorithme DQN
# Auteur : L. Barthes
# Date : 11/2025
#%%
import turtle
import matplotlib.pyplot as plt
import numpy as np
from PacmanEnvVectorSpace import Pacman, act2str
#%%
board = 'board3'       # Nom du fichier contenant la description du jeu
file = 'boards/'+board+'.txt'           # fichier contenant la description du jeu

env = Pacman(file,STEP_REWARD = -0.01,FOOD_REWARD=0.10, WIN_REWARD = 0.50, POISON_REWARD = -0.10)     # Creation de l'environnement pacman


netWeights = 'models/'+board+'.weights'   # fichier de stockage du modele neuronal

env.win.listen()
choix= input('Votre choix (manuel/train/play/stat) ? : ')
#choix='train'
#%%
if choix=='manuel':               # Mode manuel avec les touches du clavier   
    env.win.onkey(env.ActR, 'Right')
    env.win.onkey(env.ActL, 'Left')
    env.win.onkey(env.ActU, 'Up')
    env.win.onkey(env.ActD, 'Down')
    env.win.onkey(env.Quit, 'space')        # Quit
    env.win.update()
    score=0
    while not env.quit:
     if env.flag==1:
         action=env.action
         state,reward,fini,info=env.step(action) 
         env.flag=0
     env.win.update()
            
    print('Votre score est : {}'.format(score))
    turtle.done()
    turtle.bye() 
#%%    
if choix=='train':             #  apprentissage 
    from DQNPytorch_16 import DQN_Solver
    #from DDQNPytorch_16 import DQN_Solver
    env.hide()                  # De-active le graphique pour accélérer l'execution
    #env.show()                 # active le mode graphique
    env.setVerboseMode(mode=False)   # Active/De-active l'affichage du texte pour accélérer ()
    
    agent = DQN_Solver(env)        # Création de l'agent
    score = agent.train(Nbr_episodes=500,filesave=netWeights)   # On entraine le reseau et on sauve les poids
    
#%%    
if choix == 'play':           # On joue 1 coup
    import torch
    from DQNPytorch_16 import DQN_Solver
    agent = DQN_Solver(env)        # Création de l'agent
    net = agent.network
    net.eval()
    net.load_state_dict(torch.load(netWeights ))       # On charge les poids appris par train
    
    env.reset(x0=None,y0=None)        # Reset et tirage aléatoire le la position du pacman
    env.show()
    plt.pause(0.3)
    done = False
    count=0
    score_final=0
    while not done and count  < 50:         # On boucle jusqu'a fini ou on a avancé 50 fois sans gagner
        action=net.play1step(env.state())     # Le réseau détermine la prochaine action à partir de l'état courant
        print('action[{}] : {}'.format(count,act2str(action)))
        #a=input('press key')
        state,reward,done,_=env.step(action)      # On execute l'action trouvé précédemment
        score_final +=reward
        plt.pause(1)
        count +=1
    if done:
        print('vous avez gagné')
    else:
        print('vous avez perdu')
        
    print('Your final score : ',score_final,' Nombre step :',count)
    
#%%    
if choix == 'stat' or choix =='train':        # Statistiques sur 100 parties   
    
    import torch
    from DQNPytorch_16 import DQN_Solver
    agent = DQN_Solver(env)        # Création de l'agent
    net = agent.network
    net.eval()
    net.load_state_dict(torch.load(netWeights ))       # On charge les poids appris par train
    
    Npartie = 100
    scores=[]
    victoires=[]
    steps=[]
    poisons=[]
    for partie in range(Npartie):
        env.reset(x0=None,y0=None)        # Reset et tirage aléatoire le la position de la tortue
        done=False
        count=0
        score_final=0
        while not done and count  < 50:         # On boucle jusqu'a fini ou on a avancé 50 fois sans gagner
           action=net.play1step(env.state())     # Le réseau détermine la prochaine action à partir de l'état courant
           
           state,reward,done,_ =env.step(action)      # On execute l'action trouvé précédemment            score_final +=reward
           count +=1
           score_final +=reward
        scores.append(score_final)
        victoires.append(int(done))
        steps.append(count)  
        poisons.append(env.NbrePoisonEaten)
        
    print('Score moyen : ',np.mean(scores),' Pourcentage victoire :',np.round(np.sum(victoires)/Npartie*100,2), '%')   
    print('Nombre de deplacements moyens ', np.mean(steps))
    print('Nombre moyen de cases poison par partie',np.sum(poisons)/Npartie)
    print(f"board used: {board}")
#%%
turtle.done()
turtle.bye() 