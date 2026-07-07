# -*- coding: utf-8 -*-
"""
--------------------------------
 CrÃ©ation d'un environnement de jeu de pacman !
 Auteur : L. Barthes
 Date : 2023
 Version 2.0
 Espace d'état représenté par un vecteur d'état
"""
import turtle as t
import random
import matplotlib.pyplot as plt

#%%
DL0=40
posMax=0
OFFSET=-250

def PosToPix(Pos):          # Conversion coordonnÃ©e position en pixel ecran
    return round(OFFSET+Pos*DL0)
def PixToPos(Pix):          # Conversion coordonnÃ©es pixel ecran en position
    return round((Pix-OFFSET)/DL0)
def act2str(num_action):
    if num_action==0:
        return 'down'
    elif num_action==1:
        return 'left'
    elif num_action==2:
        return 'up'
    elif num_action==3:
        return 'right'
#%% Creation d'un objet graphique de type Turtle
def createItem(pos,color='red',shape='square',size=(1,1),speed=0):
     item = t.Turtle()
     item.speed(speed)
     item.shape(shape)
     item.shapesize(size[0],size[1])
     item.color(color)
     item.penup()
     item.goto(pos[0], pos[1])
     return item
#%% Lecture fichier de configuration
#Creation des objet graphiques 
class Graphic:
    def __init__(self,file,FOOD_REWARD, WIN_REWARD, POISON_REWARD, DEADLYPOISON_REWARD):
        global posMax
        
        print('Chargement board : ',file)
        self.food = []
        self.poison=[]
        self.deadlypoison=[]
        self.positions=[]
        
        with open(file) as reader:
            data=reader.read().split('\n')
            posMax= int(data[0])+1
            
            for line in data[1:]:
                l = line.split(' ')
                if len(l[0]) ==0:
                    continue
                
                T = t.Turtle()
                X=int(l[0])
                Y= int(l[1])
                #value=int(l[2])
                value=l[2]
                
                if value !='H':
                    self.positions.append((X,Y))    # on stocke la position de tous les objet excepte la maison
                    
                T.goto(PosToPix(X), PosToPix(Y))
                T.speed(0)
                T.shapesize(2,2)
                T.penup()              
                T.eat=0
                            
               
                if value =='D':
                    T.shape("images/deadlypoison.gif")
                    T.shapesize(1,1)
                    T.value= DEADLYPOISON_REWARD
                    self.deadlypoison.append(T)
                elif value =='P':
                    T.shape("images/poison.gif")
                    T.shapesize(1,1)
                    T.value= POISON_REWARD
                    self.poison.append(T)
                elif value =='F':
                    T.shape("images/food.gif")
                    T.shapesize(1,1)
                    T.value= FOOD_REWARD
                    self.food.append(T)
                elif value =='H':
                    T.shape("images/home.gif")
                    T.shapesize(1,1)
                    self.home=T
            
#%% Creation de l'environnement Pacman  
class Pacman():
    def __init__(self,file,STEP_REWARD = -1,FOOD_REWARD=10, WIN_REWARD = 50, POISON_REWARD = -20, DEADLYPOISON_REWARD=-100):   
        self.STEP_REWARD = STEP_REWARD
        self.WIN_REWARD = WIN_REWARD
        self.done = 0 #False
        self.reward = 0
        self.action=0
        self.cumulatedReward=0
        self.NbrePoisonEaten=0          # compte combien de fois on est pasé sur une case poison
        self.steps=0
        self.quit = False
        self.disp=True
        self.init_graph(file,STEP_REWARD,FOOD_REWARD,WIN_REWARD,POISON_REWARD,DEADLYPOISON_REWARD)    # Lit le fichier de configuration et créé les objet graphiques
        
        self.action_space = 4       # 4 actions possibles
        self.grid_size =posMax -1       # taille de la grille 

        self.state_space =2 + len(self.env.food) + len(self.env.poison)     # Dim du vection d'état (position+ Nbre nourrture+Nbr poison)
        self.state_space =2 + len(self.env.food)     # Dim du vection d'état (position+ Nbre nourrture)
        self.reset()
        self.flag=0
        self.verbose=True
        
        
    def show(self):
        self.disp=True
    def hide(self):
        self.disp=False
    def setVerboseMode(self,mode=True):
        self.verbose=mode
    def Xturtle(self):
        return PixToPos(self.pacman.xcor())
    def Yturtle(self):
        return PixToPos(self.pacman.ycor())
    def setXturtle(self,pos):
        self.pacman.setx(PosToPix(pos))
    def setYturtle(self,pos):
        self.pacman.sety(PosToPix(pos))    
    def ActR(self):
        self.action=3
        self.flag=1
    def ActL(self):
        self.action=1
        self.flag=1
    def ActU(self):
        self.action=2
        self.flag=1
    def ActD(self):
        self.action=0
        self.flag=1
    def Quit(self):
        self.quit=True
        self.flag=1
        
    # turtle movement
    def pacman_right(self):
        pos=self.Xturtle()
        
        if  pos< self.grid_size :
            self.setXturtle(pos+1)
        
    def pacman_left(self):
        pos=self.Xturtle()
        if pos > 1:
            self.setXturtle(pos-1)
                    
    def pacman_up(self):
        pos=self.Yturtle()
        if pos < self.grid_size:
            self.setYturtle(pos+1)
        
    def pacman_down(self):
        pos=self.Yturtle()
        if pos > 1:
            self.setYturtle(pos-1)
        
    def sample(self):             # Tire au hasard une position de la tortue
        return random.randint(0,self.action_space-1)
     
    def update_food(self):         # Mise à jour de la nourriture

        self.win.update()
        self.env.home.showturtle()
        NbreFood=0
        for food in self.env.food:
            if not self.disp:
                food.hideturtle()
                        
            if self.Xturtle() == PixToPos(food.xcor()) and self.Yturtle() == PixToPos(food.ycor()) and food.eat == 0:
                self.reward += food.value
                food.eat = 1            # Indique que la nouriture a été mangée
                food.hideturtle()
                if self.verbose :
                    print('miam miam  ...')
                #break
            if food.eat ==1:
                  NbreFood+= 1
        return NbreFood             # retourne le nombre de nourriture mangé
        
    def update_poison(self):         # mise a jour des poisons
        self.win.update()
        
        for poison in self.env.poison:
            if not self.disp:
                poison.hideturtle()
                
            if self.Xturtle() == PixToPos(poison.xcor()) and self.Yturtle() == PixToPos(poison.ycor()) and poison.eat == 0:
                self.reward +=poison.value
                
                poison.eat = 0  #1
                #poison.hideturtle()
                #self.NbrePoisonEaten +=1
                if self.verbose:
                    print('beurk pas bon  ...')
                

    def reset(self,x0=None,y0=None):
        '''
        Remise à l'etat initial et tirage aléatoire de la positon du pacman sauf si on precise sa position
        Remise à l'état non mangé de la nourriture et des poisons
        '''
        if x0 is None and y0 is None:          
            while 1==1:
                X= random.randint(1,self.grid_size)
                Y = random.randint(1,self.grid_size)
                if (X,Y) not in self.env.positions:
                    break
        else:
            X=x0
            Y=y0
            
        self.pacman.goto(PosToPix(X),PosToPix(Y))
        self.reward=0
        self.cumulatedReward=0
        self.NbrePoisonEaten=0 
       
        for food in self.env.food:
            food.eat=0
            food.showturtle()
                
        for poison in self.env.poison:
            poison.eat=0
            poison.showturtle()
            
        self.steps=0
        self.win.update()
        return self.state()         # retourne l'etat initial

    def step(self, action=None):    # fait évoluer le jeu d'un pas de temps
        '''
        Retourne l etat, le reward, done 
            done : True si partie finie sinon False
        '''
        
        self.done = 0 #False
        if self.disp:
            self.pacman.showturtle()
        else:
            self.pacman.hideturtle()
        
            
        self.reward = self.STEP_REWARD     # Reward pour un deplacement
        if action == 0:           
            self.pacman_down()
        if action == 1:
            self.pacman_left()
        if action == 2:
            self.pacman_up()
        if action == 3:
            self.pacman_right()
            
         
        self.steps+=1
        NbrFood = self.update_food()
        self.update_poison()
        self.score.clear()
        # if self.Xturtle() == PixToPos(self.env.home.xcor()) and self.Yturtle() == PixToPos(self.env.home.ycor()) :
        #     print('maison........',NbrFood)
        # Fin du jeu si pacman à la maison et plus de nourriture à manger
        if self.Xturtle() == PixToPos(self.env.home.xcor()) and self.Yturtle() == PixToPos(self.env.home.ycor()) and NbrFood == len(self.env.food):
            self.reward +=self.WIN_REWARD
            self.done= 1  #True
         
            if self.verbose:
                print('bravo vous avez gagne...')
        
        self.cumulatedReward += self.reward      # Reward cumulé pour l'affichage
        
        # affichage
        # if self.disp:
        #     self.score.write("X : {} Y : {} Reward: {:3.2f} Cum Reward: {:3.2f} step : {:4d}\nState: {}  Done :{}".format(
        #         self.Xturtle(),self.Yturtle(),self.reward,self.cumulatedReward,self.steps,self.state(),self.done), align='center', font=('Courier', 12, 'normal'))

        if self.disp:
            self.score.clear()
            list_states = [float("%2.2f " % value) for value in self.state()]
            self.score.write("X : {} Y : {} Reward: {:3.2f} Cum Reward: {:3.2f} step : {:4d}\nState: {} done : {}".format(
            self.Xturtle(),self.Yturtle(),self.reward,self.cumulatedReward,self.steps,list_states,self.done), align='center', font=('Courier', 12, 'normal'))



        return self.state(), self.reward,self.done,None

    # def state(self):               # retourne l'etat courant
        
    #     state = self.Xturtle() - 1 + self.grid_size *  (self.Yturtle() -1) 
        
    #     flagFoodEaten= [food.eat for food in self.env.food ]      # recupere liste des flags nourriture mangées/non mangées 
    #     flagPoisonEaten= [poison.eat for poison in self.env.poison ]      # recupere liste des flags nourriture mangées/non mangées 

    #     for i,flag in enumerate(flagFoodEaten+flagPoisonEaten ):
    #         state=state + flag*(self.grid_size**2)*(2**i)
    #     return state
        
    def state(self):               # retourne l'etat courant
        flagFoodEaten= [food.eat for food in self.env.food ]      # recupere liste des flags nourriture mangées/non mangées 
        flagPoisonEaten= [poison.eat for poison in self.env.poison ]      # recupere liste des flags nourriture mangées/non mangées 

        state =  [-1+2*self.Xturtle()/(self.grid_size), -1+2*self.Yturtle()/(self.grid_size)] + flagFoodEaten #+ flagPoisonEaten
        return state
    
    
    
    
    
    def init_graph(self,file,STEP_REWARD,FOOD_REWARD,WIN_REWARD,POISON_REWARD,DEADLYPOISON_REWARD):
        # Setup Background. Initialisation des objets graphiques
        self.win = t.Screen()
        self.win.title('Pacman Q-learning')
        self.win.bgcolor('black')
        self.win.setup(width=800, height=800)
        self.win.tracer(0)
       
        self.win.addshape("images/mur.gif")
        self.win.addshape("images/food.gif")
        self.win.addshape("images/poison.gif")
        self.win.addshape("images/home.gif")
        self.win.addshape("images/blank.gif")
        self.win.addshape("images/pacman.gif")
        self.win.addshape("images/deadlypoison.gif")
        
         #food
        self.env=Graphic(file,FOOD_REWARD, WIN_REWARD, POISON_REWARD,DEADLYPOISON_REWARD)
        
        # Creation du cadre jaune
        self.gauche=createItem(pos=(PosToPix(0), PosToPix(-0.5)+posMax/2*DL0),color='red',size=(posMax*2,2))
        self.droite=createItem(pos=(PosToPix(posMax), PosToPix(0.0)+posMax/2*DL0),color='red',size=(posMax*2+2,2))
        self.bas=createItem(pos=(PosToPix(0)+posMax/2*DL0, PosToPix(0.0)),color='red',size=(2,posMax*2))
        self.haut=createItem(pos=(PosToPix(-0.5)+posMax/2*DL0, PosToPix(posMax)),color='red',size=(2,posMax*2))

        # Création du pacman
        self.pacman=createItem(pos=(PosToPix(0), PosToPix(6)),shape='images/pacman.gif',size=(2,2))

        # Affichage Score
        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("X : {} Y : {} Reward: {}".format(self.Xturtle(),self.Yturtle(),self.reward), align='center', font=('Courier', 16, 'normal'))
        

#%%
if __name__ == '__main__':
   
    
    env = Pacman('boards/board0.txt')       # Creation de l'environnement
    env.win.listen()            # Mode manuel avec les touches du clavier
    import matplotlib.pyplot as plt
    env.win.onkey(env.ActR, 'Right')
    env.win.onkey(env.ActL, 'Left')
    env.win.onkey(env.ActU, 'Up')
    env.win.onkey(env.ActD, 'Down')
    env.win.onkey(env.Quit, 'space')        # Quit
    
    env.win.update()
    score=0
    #env.hide()
    while not env.quit:
     if env.flag==1:
         action=env.action
         total_reward,state,fini,info=env.step(action) 
         env.flag=0
     env.win.update()
             
     #print('{:2.1f}'.format(score))
    print('Votre score est : {}'.format(total_reward))
    t.done()
    t.bye() 
   