import pickle
from typing import Any
from cv2 import VideoCapture
from flask import Flask
from flask import render_template
from flask import Response
import cv2
import numpy as np

#Carga el modelo
model = pickle.load(open("DTmodelIris.pkl","rb"))
app = Flask(__name__)
cap = cv2.VideoCapture(0)

binaria=""
frame=""
gray=""
test_imagen=""
y=""
def generate():
    while True:
          ret, frame = cap.read()
          if ret:
               
               imagen2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
               gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
               _,binaria =  cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

               
               (contornos,_) = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
               cnt = contornos[0]
               M = cv2.moments(cnt)
               cX = int(M["m10"]/ M["m00"])
               cY = int(M["m01"]/ M["m00"])

               seplength = cX/100
               sepwidth = cY/100



               cv2.circle(frame,(cX,cY),5,(0,255,0),-1)
               cv2.putText(frame,"x:"+str(cX)+",y:"+str(cY),(cX,cY),1,1,(0,0,0),1)
               x,y,w,h=cv2.boundingRect(cnt)
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,3))
               cv2.putText(frame,"w:"+str(w)+",h:"+str(h),(w,h),1,1,(0,0,0),1)
                    
               cv2.drawContours(frame, cnt, -1, (255,0,0),2)

               cv2.imshow('Imagen', frame)
               petlength = w/1000
               petwidth = y/1000

               #Crea el arreglo con datos obtenidos de la imagen :)
               test_imagen = np.array([[seplength,sepwidth,petlength,petwidth]])

               
               cv2.waitKey(0)
               cv2.destroyAllWindows()

               y = model.predict(test_imagen)
               print(y[0])

               (flag, encodedImage) = cv2.imencode(".jpg", frame)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')





@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/predic")
def predic():
     return  y[0]



if __name__ == "__main__":
    app.run(debug=False)

