#Thư viện numpy phục vụ cho ma trận hay toán tử nhiều chiều
import numpy as np 

class Perceptron:
	def __init__(self, learning_rate=0.01, n_iters= 1000):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.activation_function = self._unit_step_function
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		#shape: kích thức của mảng
		n_sample, n_features = X.shape

		#init weights
		#tạo một numpy với tất cả phần tử là 0
		self.weights = np.zeros(n_features)
		self.bias = 0 

		y_ = np.array([1 if i >0 else 0 for i in y])

		for _ in range(self.n_iters):
			#enumerate() với input là 1 iterable và trả về là list có đánh số
			#idx là số thứ tự khi được trả về từ enumarate
			#x_i là giá trị của X
			for idx, x_i in enumerate(X):
				#np.dot trong numpy dùng để nhân 2 ma trận
				linear_output = np.dot(x_i, self.weights) + self.bias
				y_predicted = self.activation_function(linear_output)

				update = self.lr*(y_[idx] - y_predicted)
				self.weights += update*x_i
				self.bias += update


	def predict(self, X):
		linear_output = np.dot(X, self.weights) + self.bias
		y_predicted = self.activation_function(linear_output)
		return y_predicted	



	def _unit_step_function(self, x):
		return np.where(x >= 0, 1, 0)

