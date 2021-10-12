import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


archivo = pd.read_excel('analisis_curva_temperatura.xlsx',skiprows=1)
archivo = np.array(archivo)

tiempo = np.zeros(len(archivo))
temperatura = np.zeros(len(archivo))
for i in range(len(archivo)):
    tiempo[i] = archivo[i][0]
    temperatura[i] = archivo[i][1]

fun= interpolate.interp1d(tiempo,temperatura,kind='cubic')
tiempo_new=np.linspace(tiempo[0],tiempo[-1],100)
temperatura_new= fun(tiempo_new)

#grafico
y_ticks= np.arange(140,250,10)
#plt.figure(figsize=(43.2,28.8))
plt.plot(tiempo_new/60,temperatura_new)
plt.plot(12,fun(12*60),'o',label='Primer_Crack')
plt.plot(16,fun(16*60),'o',label='Segundo_Crack')
plt.title('Curva_de_tostion_220ºC')
plt.xlabel('tiempo ($min$)')
plt.xticks([i for i in range (21)])
plt.ylabel('temperatura (ºC)')
plt.yticks(y_ticks)
plt.legend()
plt.grid()
plt.savefig('Curva_de_tostion_220ºC.png')


#analisis imagenes
import skimage
import os
from skimage import io
from skimage.segmentation import slic

filename = os.path.join(skimage.data_dir, 'cafe_1.png')
cafe_1 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_2.png')
cafe_2 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_3.png')
cafe_3 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_4.png')
cafe_4 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_5.png')
cafe_5 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_6.png')
cafe_6 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_7.png')
cafe_7 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_8.png')
cafe_8 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_9.png')
cafe_9 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_10.png')
cafe_10 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_11.png')
cafe_11 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_12.png')
cafe_12 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_13.png')
cafe_13 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_14.png')
cafe_14 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_15.png')
cafe_15 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_16.png')
cafe_16 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_17.png')
cafe_17 = io.imread(filename)

filename = os.path.join(skimage.data_dir, 'cafe_18.png')
cafe_18 = io.imread(filename)

color_1=np.arange(0,18,1)
color_2=np.arange(0,18,1)
color_3=np.arange(0,18,1)



mean_18=np.average(cafe_18,axis=0)
mean_18=np.average(mean_18,axis=0)

print(len(mean_18))
