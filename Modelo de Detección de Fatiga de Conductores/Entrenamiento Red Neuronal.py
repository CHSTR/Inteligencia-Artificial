import numpy as np
import tensorflow as tf
import tflearn
import pandas as pd
##JOSE RAMIREZ
##CHRISTOPHER STEARS
#Se leen los datos
datos = pd.read_csv('datos_entrenamiento.csv', encoding='latin-1')
#Se desordenan
datos = datos.iloc[np.random.permutation(len(datos))]
print(datos.head())
print(datos.shape)
#index = ['Posición del rostro,Apertura de Ojos,Posición de las manos,Pulso,Ruta,Posición torso,Uso de pedales,Tiempo de descanso,Distancia con otros vehículos, Alerta\n]
#Se guardan los valores en x e y
x = datos.iloc[:,:-1].values
y = datos.iloc[:, -1].values
#Se ajusta la dimension y modificamos los datos a [1,0] o [0,1]
y = np.array(y)
y = y.reshape(len(y),1)
print(y)
T = []
print("asd")
for i in y:
    if i == 1:
        T.append([1,0,0]) #ALERTA
    elif i == 2:
        T.append([0,1,0]) #Estado de fatiga
    else:
        T.append([0,0,1]) # no hay alerta
#Se guardan los datos de prueba
print("asd")
test = pd.read_csv('test.csv')
print(test.shape)
#Se guardan los valores
prueba = test.iloc[:,:].values
print(prueba)
#Se definen los datos de entradas, salida y de las capas
entradas = 9
capa1 = 6
salida = 3
#funcion que crea el modelo de la red
def modelo():
    tf.reset_default_graph()
    red = tflearn.input_data([None, entradas])
    red = tflearn.fully_connected(red, capa1, activation='sigmoid')
    #red = tflearn.fully_connected(red, capa2, activation='sigmoid')
    #Softmax, distribucion de prob. que modifica las salidas a valores entre [0,1]
    red = tflearn.fully_connected(red, salida, activation='softmax')
    #Se define el optimizador el "Gradiente estocastico" con una medida de error "Cuadratico Medio"
    red = tflearn.regression(red, optimizer='sgd', loss='mean_square', learning_rate=0.1)
    #Se guarda la red en la variable
    modelo = tflearn.DNN(red)
    return modelo
    
modelo_red_neuronal = modelo()
#Se comeinza a ajustar la red neuronal
modelo_red_neuronal.fit(x,T, validation_set=0.5, n_epoch=20, show_metric=True)
#Resultados
print(modelo_red_neuronal.predict(prueba))
modelo_red_neuronal.save('Final-AI')

