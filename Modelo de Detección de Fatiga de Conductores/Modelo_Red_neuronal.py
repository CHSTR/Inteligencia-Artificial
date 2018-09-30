import numpy as np
import tensorflow as tf
import tflearn
import pandas as pd
import os
##JOSE RAMIREZ
##CHRISTOPHER STEARS

entradas = 9
capa1 = 6
salida = 3
#funcion que crea el modelo de la red
def modelo():
    tf.reset_default_graph()
    red = tflearn.input_data([None, entradas])
    #Funcion de activacion ReLU (si x>0: x, de lo contrario: 0)
    red = tflearn.fully_connected(red, capa1, activation='sigmoid')
    #red = tflearn.fully_connected(red, capa2, activation='sigmoid')
    #Softmax, distribucion de prob. que modifica las salidas a valores entre [0,1]
    red = tflearn.fully_connected(red, salida, activation='softmax')
    #Se define el optimizador el "Gradiente estocastico" con una medida de error "Cuadratico Medio"
    red = tflearn.regression(red, optimizer='sgd', loss='mean_square', learning_rate=0.1)
    #Se guarda la red en la variable
    modelo = tflearn.DNN(red)
    return modelo
    
modelo_red = modelo()
if os.path.exists('C:/Users/crist/Desktop/Final AI/{}.meta'.format('Final-AI')): #C:/CARPETA/.../
    modelo_red.load('Final-AI')
    print('MODELO CARGADO!')

while 1:
    T=[]
    a = input('Posición del rostro: -1: izq, 0: normal, 1: derecha\n')
    T.append(a)
    a = input('Apertura de Ojos: 0: cerrados, 1: abiertos\n')
    T.append(a)
    a = input('Posición de las manos: 0: sin manos al volante, 1: con 1, 2: con 2\n')
    T.append(a)
    a = input('Pulso: -1: bajo lo normal, 0: normal, 1: acelerado\n')
    T.append(a)
    a = input('Ruta: 0: en ruta, 1: fuera de ruta\n')
    T.append(a)
    a = input('Posición torso: 0: relajado, 1: concentrado, 2: tenso, 3: cansado\n')
    T.append(a)
    a = input('Uso de pedales: 0: no hace uso, 1: hace uso\n')
    T.append(a)
    a = input('Tiempo de descanso: 0: menos de 8 horas, 1: mayor o igual a 8 horas\n')
    T.append(a)
    a = input('Distancia con otros vehículos: 0: muy cerca , 1: distancia prudente\n')
    T.append(a)
    T = np.array(T).reshape(1,9)
    print(T)
    salida = modelo_red.predict(T)
    if np.argmax(salida) == 2:
        print("Predicción de la Red Neuronal: {}\n".format(modelo_red.predict(T)))
        print("NO HAY ALERTA\n")
    elif np.argmax(salida) == 1:
        print("Predicción de la Red Neuronal: {}\n".format(modelo_red.predict(T)))
        print("RIESGO DE FATIGA\n")
    else:
        print("Predicción de la Red Neuronal: {}\n".format(modelo_red.predict(T)))
        print("HAY ALERTA\n")
    


