import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets #datasets là bộ dữ liệu được chuẩn hóa trong thư viện 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


from perceptron  import  Perceptron

X, y = datasets.make_blobs(n_samples = 100, n_features = 3, centers = 2, cluster_std = 1.1, random_state = None )


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

p = Perceptron(learning_rate = 0.01, n_iters = 1000)
p.fit(X_train, y_train)


def test(y_true, y_pred):
	acc =  np.sum(y_true == y_pred)/ len(y_true)
	return acc

predictions = p.predict(X_test)
accu = (test(y_test, predictions) * 100)


ax = plt.axes(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c= y_train , cmap='viridis', linewidth=0.5);

x0_1 = np.amin(X_train[:,0])
x0_2= np.amax(X_train[:,0])
x0_3= X_train[0][0]


xx =[x0_1,x0_2,x0_3]

x1_1 = np.amin(X_train[:,1])
x1_2= np.amax(X_train[:,1])
x1_3= X_train[1][1]

yy =[x1_1,x1_2,x1_3]

xx, yy = np.meshgrid(xx, yy)

x2_1 = (-p.weights[0] *x0_1 -p.weights[1]*x1_1 - p.bias)/p.weights[2]
x2_2 = (-p.weights[0] *x0_2 -p.weights[1]*x1_2 - p.bias)/p.weights[2]
x2_3= (-p.weights[0] *x0_3 -p.weights[1]*x1_3 - p.bias)/p.weights[2]

zz = (-p.weights[0] *xx -p.weights[1]*yy - p.bias)/p.weights[2]

surf = ax.plot_surface(xx, yy, zz, linewidth=0, antialiased=False)

ax.set_title("PERCEPTRON 3D TEST")
ax.set_xlabel("X1 FEATURE")
ax.set_ylabel("X2 FEATURE")
ax.set_zlabel("X3 FEATURE")
ax.legend(title = 'Accuracy: %0.3f percent'%accu)

plt.show()