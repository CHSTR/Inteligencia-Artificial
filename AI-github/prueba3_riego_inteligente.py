import numpy as np
import tensorflow as tf
import tflearn
import pandas as pd
##JOSE RAMIREZ
##CHRISTOPHER STEARS
#Se leen los datos
datos = pd.read_csv('datos_entrenamiento.csv')
#Se desordenan
datos = datos.iloc[np.random.permutation(len(datos))]
print(datos.head())
print(datos.shape)
#index = ['TemperaturaAmbiente','HumedadActual','TipoSuelo','PhAgua','HumedadDeseada','TemperaturaDeseadaSuelo','Regaras']
#Se guardan los valores en x e y
x = datos.iloc[:,:-1].values
y = datos.iloc[:, -1].values
#Se ajusta la dimension y modificamos los datos a [1,0] o [0,1]
y = np.array(y)
y = y.reshape(len(y),1)
T = []
for i in y:
    if i == 1:
        T.append([1,0]) #regar
    else:
        T.append([0,1]) #no regar
#Se guardan los datos de prueba
test = pd.read_csv('test.csv')
#Se guardan los valores
prueba = test.iloc[:,:].values
#Se definen los datos de entradas, salida y de las capas
entradas = 6
capa1 = 6
capa2 = 6
salida = 2
#funcion que crea el modelo de la red
def modelo():
    tf.reset_default_graph()
    red = tflearn.input_data([None, entradas])
    #Funcion de activacion ReLU (si x>0: x, de lo contrario: 0)
    red = tflearn.fully_connected(red, capa1, activation='ReLU')
    red = tflearn.fully_connected(red, capa2, activation='ReLU')
    #Softmax, distribucion de prob. que modifica las salidas a valores entre [0,1]
    red = tflearn.fully_connected(red, salida, activation='softmax')
    #Se define el optimizador el "Gradiente estocastico" con una medida de error "Cuadratico Medio"
    red = tflearn.regression(red, optimizer='sgd', learning_rate=0.1, loss='mean_square')
    #Se guarda la red en la variable
    modelo = tflearn.DNN(red)
    return modelo
    
modelo_red_neuronal = modelo()
#Se comeinza a ajustar la red neuronal
modelo_red_neuronal.fit(x,T, validation_set=0.2, n_epoch=15)
#Resultados
print(modelo_red_neuronal.predict(prueba))

