from keras.layers import Dense
from keras.models import Sequential
import random
import matplotlib.pyplot as plt
import math
gurultu = 0.2
X = []
y = []
for i in range(0,1000):
    angle=random.uniform(-math.pi,math.pi)
    X.append(angle)
    y.append(math.sin(angle)+random.uniform(-gurultu,gurultu))

plt.scatter(X,y,s=0.1)
plt.xlabel('x (Radyan)')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
model = Sequential()
model.add(Dense(100, input_shape=(1,), activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd')
history=model.fit(X, y, epochs=50, verbose=1)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.grid()
plt.show()
X_test = []
y_test = []
for i in range(-1800,1800):
    angle = math.radians(i/10)
    X_test.append(angle)
    y_test.append(math.sin(angle))

def testmodel(X,y):
    res = model.predict(X, batch_size=256)
    plt.plot(X,y, label='sin')
    plt.plot(X,res, label='sonuc')
    plt.xlabel('x (Radyan)')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()

testmodel(X_test,y_test)
girilen=float(input("Radyan Değerini Griniz= "))
y_proba = model.predict([girilen])
print("Gerçek Sinüs Değeri= ",math.sin(girilen))
print("Makinenin Öğrendiği Sinüs Değeri= ",y_proba)