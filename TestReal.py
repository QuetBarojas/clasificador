import pickle
import numpy as np
import cv2 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#Carga el modelo
model = pickle.load(open("DTmodelIris.pkl","rb"))

#carga la imagen
sns.set (color_codes=True)

imagen = cv2.imread('petalo4.jpg')
imagen2 = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
grises = cv2.cvtColor(imagen2,cv2.COLOR_BGR2GRAY)
_,binaria =  cv2.threshold(grises, 225, 255, cv2.THRESH_BINARY_INV)
np.array(binaria)

#--fin carga de imagen

# Obtiene datos de la imagen

(contornos,_) = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contornos[0]
M = cv2.moments(cnt)
cX = int(M["m10"]/ M["m00"])
cY = int(M["m01"]/ M["m00"])

seplength = cX/100
sepwidth = cY/100



cv2.circle(imagen,(cX,cY),5,(0,255,0),-1)
cv2.putText(imagen,"x:"+str(cX)+",y:"+str(cY),(cX,cY),1,1,(0,0,0),1)
x,y,w,h=cv2.boundingRect(cnt)
cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,3))
cv2.putText(imagen,"w:"+str(w)+",h:"+str(h),(w,h),1,1,(0,0,0),1)
       
cv2.drawContours(imagen, cnt, -1, (255,0,0),2)

cv2.imshow('Imagen', imagen)
petlength = w/1000
petwidth = y/1000

#Crea el arreglo con datos obtenidos de la imagen :)
test_imagen = np.array([[seplength,sepwidth,petlength,petwidth]])

cv2.drawContours(imagen,contornos,-1,(0,0,255), 2)
cv2.imshow("contornos", imagen)
cv2.waitKey(0)

    
cv2.waitKey(0)
cv2.destroyAllWindows()

#Realizar la predicción

y = model.predict(test_imagen)

print(y[0])
