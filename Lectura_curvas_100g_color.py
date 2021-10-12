import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#prueba_1 no valida
#prueba_1=np.load('prueba_4_julio_1_1.npy')
#prueba_1=prueba_1[1:1900]
#prueba_2
prueba_2=np.load('prueba_4_julio_1_2.npy')
prueba_2=prueba_2[1:1520]
#prueba_3
prueba_3=np.load('prueba_4_julio_3.npy')
prueba_3=prueba_3[1:1400]
#prueba_4
prueba_4=np.load('prueba_4_julio_4.npy')
prueba_4=prueba_4[1:1400]
#prueba_5
prueba_5=np.load('prueba_4_julio_5.npy')
prueba_5=prueba_5[1:1310]
#prueba_6
prueba_6=np.load('prueba_4_julio_6.npy')
prueba_6=prueba_6[1:1310]
#prueba_7
prueba_7=np.load('prueba_4_julio_7.npy')
prueba_7=prueba_7[150:1270]
#prueba_8
prueba_8=np.load('prueba_4_julio_8.npy')
prueba_8=prueba_8[74:970]
#prueba_9
prueba_9=np.load('prueba_4_julio_9.npy')
prueba_9=prueba_9[50:780]
#prueba_10
prueba_10=np.load('prueba_4_julio_10.npy')
prueba_10=prueba_10[1:720]

pruebas=np.array([prueba_2,prueba_3,prueba_4,prueba_5,prueba_6,prueba_7,prueba_8,prueba_9,prueba_10])
index=np.arange(0,len(pruebas),1)
temperaturas_finales=[]

for prueba in pruebas:
    prueba=gaussian_filter1d(prueba,50)
    time=np.linspace(0,len(prueba),len(prueba))*0.2/60
    plt.plot(time,prueba)
    plt.plot(time[-1],prueba[-1],'o')
    temperaturas_finales.append(prueba[-1])
plt.title('Temperature_vs_time_100_grams')
plt.xlabel('Time_($min$)')
plt.ylabel('Temperature_($ºC$)')
plt.grid()
plt.savefig('pruebas_100.pdf')
plt.show()


temperaturas_finales=np.array(temperaturas_finales)
temperaturas_finales=temperaturas_finales.reshape(-1,1)
#perdida de peso
peso=np.array([70,79,77,79,77,80,83,86,86])/100

#Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(temperaturas_finales,peso)

# Make predictions using the testing set
temperatures_y_pred = regr.predict(temperaturas_finales)

plt.plot(temperaturas_finales,peso,'o')
plt.plot(temperaturas_finales,temperatures_y_pred)
plt.xlabel('Temperature_($ºC$)')
plt.ylabel('$Coffee/(Green-Coffee)$')
plt.grid()
plt.savefig('perdida_peso_cafe.pdf')
plt.show()
