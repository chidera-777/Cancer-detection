import numpy as np
import copy, math 
from collections import deque


def accuracy_score(y, y_pred):
    res = np.mean(y_pred == y) * 100
    print(f"Accuracy Score: {res}")
    

class LogisticRegression:
    
    def __init__(self):
        pass
    
    def sigmoid(self, z):
        """
        Function to compute the sigmoid of z

        Args:
            z : ndarray of any size
            
        Returns:
            g: sigmoid(z)
        """
        z = np.clip( z, -500, 500)
        g = 1.0/(1.0+np.exp(-z))
        return g
    
    
    def compute_cost(self, X, y, w, b, lambda_=1, safe=False):
        """
        Function for the computing the cost function of the model
        
        Args:
            X (ndarray (m,n)): m examples with n features
            y (ndarray (m,)): target values
            w (ndrray): weight parameter for the model
            b (scalar) : bias parameter for the model
            lambda_ (int): controls regularization. Defaults to 1.
            
        Returns:
            total_cost: cost_function of the model
        """
        
        m, n = X.shape
        cost = 0.0
        reg_cost = 0
        epsilon = 1e-10
        
        for i in range(m):
            z = np.dot(X.iloc[i], w) + b 
            func_wb = self.sigmoid(z)
            cost += -np.sum(y.iloc[i] * np.log(np.clip(func_wb, epsilon, 1 - epsilon)) + (1 - y.iloc[i]) * np.log(np.clip(1 - func_wb, epsilon, 1 - epsilon)))
                
        cost = cost/m
        
        for j in range(n):
            reg_cost += w[j]**2
        reg_cost = (lambda_/(2*m)) * reg_cost
        
        total_cost = cost + reg_cost
        
        return total_cost

    def compute_gradient(self, X, y, w, b, lambda_):
        """
        Function to compute the gradient cost of the model

        Args:
            X (ndarray (m,n)): m examples with n features
            y (ndarray (m,)): target values
            w (ndrray): weight parameter for the model
            b (scalar) : bias parameter for the model
            lambda_ (int): controls regularization. Defaults to 1.
        
        Returns:
            dj_dw (ndarray): the gradient function of the model for the weight parameter
            dj_db (ndarray): the gradient function of the model for the bias parameter
        """
        
        m,n = X.shape
        
        cost_w = np.zeros((n,))
        cost_b = 0.0
        
        for i in range(m):
            func_wb = self.sigmoid(np.dot(X.iloc[i], w) + b)
            cost_b += func_wb - y.iloc[i]
        
            for j in range(n):
                cost_w[j] += (func_wb - y.iloc[i]) * X.iloc[i, j]
                
        dj_db = cost_b/m
        dj_dw = cost_w/m
        
        for j in range(n):
            dj_dw[j] += (lambda_/m) * w[j]
        
        return dj_db, dj_dw
    
    
    def gradient_descient(self, num, X, y, w_in, b_in, alpha, lambda_):
        """
        Function to calculate the gradient descient of the model

        Args:
            X (ndarray (m,n)): m examples with n features
            y (ndarray (m,)): target values
            w (ndrray): weight parameter for the model
            b (scalar) : bias parameter for the model
        """
        J_history = deque()
        w = copy.deepcopy(w_in)
        b = b_in
        
        for i in range(num):
            
            dj_db, dj_dw = self.compute_gradient(X, y, w, b, lambda_)
            w = w - (alpha * dj_dw)
            b = b - (alpha * dj_db)
            
            if i<10000:
                J_history.append(self.compute_cost(X, y, w, b, lambda_))
            
            if i % math.ceil(num / 10) == 0:
                print(f"Number of Iterations: {i}, Cost {J_history.pop()}")
                
        return w, b, J_history
    
    def train_data(self, X, y):
        """
        A function to train the model

        Args:
            X ((m, n), ndarray): Training examples with features n
            y ((m), ndarray): Target Prediction
        """
        
        if X.ndim == 1:
            w_in = 0.1
        else:
            w_in = np.full((X.shape[1]), 0.1)
        b_in = 0.1
        alpha = 6e-5
        lambda_ = 1
        iteration = 1000
        self.weight, self.bias, Cost_history = self.gradient_descient(iteration, X, y, w_in, b_in, alpha, lambda_)
        print(f"Weight: {self.weight}, Bias: {self.bias}")
      
    def predict(self, X):
        """
        A Function to predict the target output with the trained model

        Args:
            X ((m, n), ndarray): Training examples with features n
        """
        if X.ndim == 1:
            m = X.shape
        else:
            m, n = X.shape
        prediction = np.zeros(m)
        
        for i in range(m):
            z = 0
            for j in range(n):
                z += np.dot(X.iloc[i, j], self.weight[j])
                
            z += self.bias
            
            func_wb = self.sigmoid(z)
            if func_wb >= 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
                
        return prediction
            
            
            