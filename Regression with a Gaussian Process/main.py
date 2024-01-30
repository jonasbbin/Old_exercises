
#Generic Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#Used for ridge regression
from sklearn.linear_model import Ridge

#Used for kernels in the gaussian process kernels and GP itself
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor

#Nystroem approximation, quiet bad here in this task
from sklearn.kernel_approximation import Nystroem

#norm used for outlier detection
from scipy.stats import norm
#Not really used atm
from sklearn.svm import SVR

#Backward feature seleciton
from sklearn.feature_selection import RFE

#Forward feature seleciton
from sklearn.feature_selection import GenericUnivariateSelect

#Removes constant/zero features
from sklearn.feature_selection import VarianceThreshold

#Standardize data compute z score
def standardize_data(inputMatrix):
    
    for i in inputMatrix:
        inputMatrix[i] = (inputMatrix[i] - inputMatrix[i].mean()) / inputMatrix[i].std(ddof=0)
        inputMatrix[i] = inputMatrix[i].fillna(0)
    return inputMatrix
    
    
# Reading in data
train_x = pd.read_csv('X_train.csv')
train_y = pd.read_csv('y_train.csv')
yraw = train_y['y']

#print(train_x)

train_x = train_x.drop(['id'], axis=1)
train_y = train_y.drop(['id'], axis=1)

#Standardize Data and input missing values
train_x = standardize_data(train_x)


train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

test_x = pd.read_csv('X_test.csv')
index = np.array(test_x[['id']]).astype(int)


test_x = test_x.drop(['id'], axis=1)

test_x = standardize_data(test_x)

test_x = test_x.to_numpy()

abs = np.absolute(train_x)

arr = []
for i in abs:
    arr.append(i.sum())

arr = np.array(arr)
arr = np.argsort(arr)

arr = arr[np.arange(1100)]

xTrain = []
yTrain = []

for i in arr:
    xTrain.append(train_x[i])
    yTrain.append(train_y[i])

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

#Here comes our training model
class Model(object):


    def __init__(self):
        
        #These are the model we are currently supporting, Ridgeregression or GPR with an RBF or Matern kernel
        self.mod = Ridge(alpha=1)
        #kernel = Matern(length_scale=20, length_scale_bounds= "fixed", nu=1.5)
        #kernel = RationalQuadratic(length_scale=15, length_scale_bounds="fixed", alpha=1.5)
        kernel = RationalQuadratic(length_scale=16, length_scale_bounds="fixed", alpha=1.5)
        #kernel = 1.0 * RBF(length_scale=3, length_scale_bounds="fixed") 
        self.gpr = GaussianProcessRegressor(kernel=kernel)


        #There are different supported feature maps
        #self.feature_map = Nystroem(n_components=200)
        #This was used for something but not anymore
        #self.estimator = SVR(kernel="linear")
        #This is used for backward features selection but very slow, not working atm
        #self.selector = RFE(self.estimator, n_features_to_select=100, step=2)
        #Forward feature selection
        #self.transformer = GenericUnivariateSelect(mode='k_best', param=215)
        self.transformer = GenericUnivariateSelect(mode='k_best', param=212)
       

    def predict(self, x: np.ndarray) -> np.ndarray:
        

        gp_mean = np.zeros(x.shape[0], dtype=float)
        gp_std = np.zeros(x.shape[0], dtype=float)
        predictions = np.zeros(x.shape[0], dtype=float)

        #Removes constant features (needed)
        x = self.select.transform(x)
        #Forward feature selection
        x_transformed = self.transformer.transform(x)
        #For lasso feature selection
        #x_transformed = self.mode.transform(x)
        #For Nystroem feature selection (bad)
        #x_transformed = self.feature_map.transform(x)
        
        gp_mean= self.gpr.predict(x_transformed)

        
        print("Predicting 1 finished")
        for i in range(len(gp_mean)):
            predictions[i] = gp_mean[i]

        return predictions
    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):

        #Removes constant features (needed)
        self.select = VarianceThreshold()
        self.select = self.select.fit(train_x)
        train_x = self.select.transform(train_x)


        #self.selector = self.selector.fit(train_x, train_y.ravel())

        #Forward feature selection
        self.transformer = self.transformer.fit(train_x, train_y.ravel()) 
        train_x_transformed = self.transformer.transform(train_x)

        #Nystroem feature selection/reduction (bad)
        #train_x_transformed = self.transform(train_x)
        #train_x_transformed = self.feature_map.fit_transform(train_x)


        #This uses Lasso or SVC regression for feature selection
        #lsvc = linear_model.Lasso().fit(train_x, train_y.ravel())
        #lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(train_x, train_y.ravel())
        #self.mode = SelectFromModel(lsvc, prefit=True)
        #train_x_transformed = self.mod.transform(train_x)
       
        #GPR training
        self.gpr = self.gpr.fit(train_x_transformed,train_y)

        #Ridge regression fitting
        #self.mod.fit(train_x_transformed, train_y)
        


#Calling the trainig and testing and outputing it
print('Fitting model')
model = Model()
model.fit_model(train_x, train_y)

# Predict on the test features
print('Predicting on test features')
sol = model.predict(test_x)
sol = np.array(sol)

data = {'id': index.flatten(),
        'y': sol.flatten()}

df = pd.DataFrame(data)
df.to_csv(r"E:\AML\Task1\AML1alternate.csv",index = False, header=True)


