# -*- coding: utf-8 -*-
'''--------------------------------
# jeu de PACMAN version Qlearning
Introduction à l'apprentissage par renforcement'
# Auteur : L. Barthes
# Date : 12/2025
'''
import turtle
import matplotlib.pyplot as plt
import numpy as np
from PacmanEnvDiscreteSpace import Pacman, act2str

board = 'board2'                        # Nom du fichier contenant la description du jeu
Qfile='qtables/'+board+ '.npy'          # file de sauvegarde de la table Q
file = 'boards/'+board+'.txt'           # fichier contenant la description du jeu

env = Pacman(file,STEP_REWARD = -1, FOOD_REWARD=10, WIN_REWARD = 50, POISON_REWARD = -10)     # Creation de l'environnement pacman (-1,10, 50,-10)
env.win.listen()

choix= input('Votre choix (manuel/train/play/stat) ? : ')


if choix=='manuel':               # Mode manuel avec les touches du clavier   
    env.win.onkey(env.ActR, 'Right')
    env.win.onkey(env.ActL, 'Left')
    env.win.onkey(env.ActU, 'Up')
    env.win.onkey(env.ActD, 'Down')
    env.win.onkey(env.Quit, 'space')        # Quit
    
    env.win.update()
    
    while not env.quit:     # barre d'espace pour quitter
     if env.flag==1:
         action=env.action
         state,reward,fini,info=env.step(action) 
         env.flag=0
     env.win.update()
            
    turtle.done()
    turtle.bye() 
    
if choix=='train':             #  apprentissage avec l'algo Q learning 
    from Qlearning import Qlearning
    env.hide()                  # De-active le graphique pour accélérer l'execution
    #env.show()                   # Active l'affichage
    env.setVerboseMode(mode=False)   # Active/De-active l'affichage du texte pour accélérer l'éxecution
    
    # Apprentissage de la table
    
    Q_table = Qlearning(env,env.action_space,env.state_space,
                        total_episode=1000,
                        lr=0.08,
                        gamma=0.95,
                        exploration_decreasing_decay=0.1,
                        max_steps = 150 ,
                        min_exploration_proba = 0.01,
                        board_name = board)        # Estimation de la table Q
    # Sauvegarde de la table
    np.save(Qfile,Q_table)
    print(np.round(Q_table,2))          # Affichage table Q
    
   
    
if choix == 'play':           # Joue une partie avec la table Q apprise (50 coups max)

    Q_table = np.load(Qfile)        # On charge la table Q
    
    env.reset(x0=None,y0=None)        # Reset et tirage aléatoire le la position de pacman
    fini=False
    count=1
    score_final=0
    while not fini and count  < 50:         # On boucle jusqu'a fini ou on a avancé 50 fois sans gagner
        action=np.argmax(Q_table[env.state(),:])     # on choisit l'action qui a le meilleur reward cumulé
        print('action[{}] : {}'.format(count,act2str(action)))
        
        state,reward,fini,gagne=env.step(action)      # On execute l'action trouvé précédemment
        score_final +=reward
        print(reward,score_final)
        plt.pause(0.5)
        count +=1
    print('Your final score : ',score_final,' Nombre step :',env.steps)
    print('Vous avez mangé ',env.NbrePoisonEaten,' poisons !')
    turtle.done()
    turtle.bye() 
    
if choix == 'stat' or choix=='train':           # Calcule des stat a partir de la table Q apprise

    Q_table = np.load(Qfile)        # On charge la table Q
    
    Npartie = 100            # Stat sur 100 parties
    print('Calcul des stat sur {} parties'.format(Npartie))
    scores=[]
    victoires=[]
    steps=[]
    poisons=[]
    for partie in range(Npartie):
        env.reset(x0=None,y0=None)        # Reset et tirage aléatoire de la position de pacman
        fini=False
        count=1
        score_final=0
        while not fini and count  < 50:         # On boucle jusqu'a fini ou on a avancé 50 fois sans gagner
            action=np.argmax(Q_table[env.state(),:])     # on choisit l'action qui a le meilleur reward cumulé
            state,reward,fini,gagne=env.step(action)      # On execute l'action trouvé précédemment
            score_final +=reward
            count +=1
        scores.append(score_final)
        victoires.append(int(fini))
        steps.append(count)  
        poisons.append(env.NbrePoisonEaten )
        
    print('Récompense cumulée moyenne : ',np.mean(scores),' Pourcentage victoire :',np.round(np.sum(victoires)/Npartie*100,2), '%')   
    print('Nombre de deplacements moyens ', np.mean(steps))
    print('Nombre moyen de cases poison par partie',np.round(np.sum(poisons)/Npartie,2))
   
turtle.done()
turtle.bye() 