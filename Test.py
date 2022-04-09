import pickle
import numpy as np

model = pickle.load(open("DTmodelIris.pkl","rb"))

test = np.array([[3.99,2.99,0.8,0.0]])
test2 = np.array([[5.7, 2.8, 4.5, 1.3]])

y = model.predict(test2)

print(y)
