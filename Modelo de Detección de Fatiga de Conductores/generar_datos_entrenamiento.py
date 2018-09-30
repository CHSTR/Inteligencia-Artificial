import pandas as pd
import random
from random import randint

def entrenar_0():
    with open('datos_entrenamiento.csv','w') as f:
        f.write('Posición del rostro,Apertura de Ojos,Posición de las manos,Pulso,Ruta,Posición torso,Uso de pedales,Tiempo de descanso,Distancia con otros vehículos, Alerta\n')
            
    with open('datos_entrenamiento.csv','a') as f:
        for i in range(5000):
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(randint(-1,1),1,randint(1,2),0,0,randint(0,1),1, 1,1,0))


def entrenar_01():            
    with open('datos_entrenamiento.csv','a') as f:
        for i in range(2500):
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(-1,1,1,randint(0,1),randint(0,1),randint(2,3), 1, 0,0,2))
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(1,1,1,randint(0,1),randint(0,1),randint(2,3), 1, 0,0,2))


def entrenar_1():            
    with open('datos_entrenamiento.csv','a') as f:
        for i in range(2500):
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(-1,0,0,-1,1,3, 0, 0,randint(0,1),1))
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(1,0,0,randint(0,1),1,3, 0, 0,randint(0,1),1))

entrenar_0()
entrenar_01()
entrenar_1()
