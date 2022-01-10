#Thư viện numpy phục vụ cho ma trận hay toán tử nhiều chiều
import numpy as np 

class Perceptron:
	def __init__(self, learning_rate=0.01, n_iters= 1000):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.activation_function = self._unit_step_function
		#weights: w1, w2....
		self.weights = None
		#w0 ~ bias
		self.bias = None

		#training sample: X, training label: y
	def fit(self, X, y):
		#shape: kích thức của mảng
		#ma trận X size M x n với M là hàng, với số lượng sample, n là cột, với số lượng feature 
		n_sample, n_features = X.shape

		#init weights
		#tạo một numpy với tất cả phần tử là 0, ở đây số lượng w = với số feature ~ số trục x1 x2 x3...
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
		#đầu tiên thực hiện hàm tuyến tính f(w,x) = (w^T*x +bias)
		linear_output = np.dot(X, self.weights) + self.bias
		#tiếp theo thực hiện hàm output là activate function(linear function)
		y_predicted = self.activation_function(linear_output)
		return y_predicted	



	def _unit_step_function(self, x):
		#xét toàn bộ mảng x nếu có giá trị lớn hơn 0, true thì trả về 1 và false thì trả về 0 (step fucntion)
		return np.where(x >= 0, 1, 0)

