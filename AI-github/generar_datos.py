import pandas as pd
import random
from random import randint

def entrenar_0():
    with open('datos_entrenamiento.csv','w') as f:
        f.write('TemperaturaAmbiente,HumedadActual,TipoSuelo,PhAgua,HumedadDeseada,TemperaturaDeseadaSuelo,Regaras\n')
            
    with open('datos_entrenamiento.csv','a') as f:
        for i in range(500):
            f.write('{},{:.2f},{},{},{:.2f},{},{}\n'.format(randint(0,15),random.uniform(0.6, 1.0),1,randint(6,9),random.uniform(0.0,0.5),randint(0,14),0))

def entrenar_1():            
    with open('datos_entrenamiento.csv','a') as f:
        for i in range(500):
            f.write('{},{:.2f},{},{},{:.2f},{},{}\n'.format(randint(16,35),random.uniform(0.0, 0.5),1,randint(6,9),random.uniform(0.6,1.0),randint(14,30),1))

entrenar_0()
entrenar_1()
