import numpy as np
import sklearn.svm as svm
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel
import pandas as pd

def standardize_data(data, nrRows, nrCol):
    
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    
    for i in range(0, nrRows, 12):
        for j in range(3, nrCol, 1):
            for p in range(12):
                if np.isnan(data[i+p,j]):
                    data[i+p,j] = 0
                else:
                    data[i+p,j] = (data[i+p,j] - mean[j]) / std[j]
    
    return data

def create_feature_matrix(X, num_patients, num_labels):
    X_new = np.zeros((int(num_patients/12),171))
    for i in range(0, num_patients, 12):
        patient_features = np.zeros(171)
        patient_features[0] = X[i,2]
        for j in range(3,num_labels,1):
            patient_features[(j-3)*5+1] = np.mean(X[i:i+12,j])
            patient_features[(j-3)*5+2] = np.min(X[i:i+12,j])
            patient_features[(j-3)*5+3] = np.max(X[i:i+12,j])
            
            first_non_nan = 0
            first = 0
            last_non_nan = 0
            last = 0
            dist = 0
            
            for p in range(12):
                if first_non_nan == 0:
                    first_non_nan = X[i+p,j]
                    first = p
                if X[i+p,j] != 0:
                    last_non_nan = X[i+p,j]
                    last = p
                
            dist = last - first
            if dist == 0:
                patient_features[(j-3)*5+4] = 0
            else:
                patient_features[(j-3)*5+4] = (last_non_nan - first_non_nan) / dist
            
            patient_features[(j-3)*5+5] = np.std(X[i:i+12,j])

        X_new[int(i/12)] = patient_features
        
    return X_new

def sigmoid(x):
    return 1/(1 + np.exp(-x))
sigmoid_vec = np.vectorize(sigmoid)

inDEV = True

# ============================================================================
# PREPROCESSING
# ============================================================================

# nr of raw training data (12 measurements per patient)
nrTrainingRaw = 227940
# nr of patiens
nrPatientsTrain = 18995
# nr of test data
nrTestRaw = 151968
# nr of patients in test data
nrPatientsTest = 12664

# nr of features (with pid)
nrFeatures = 37
# nr of labels (with pid)
nrLabels = 16

# variables to store training and test data
X = np.zeros((nrTrainingRaw, nrFeatures))
y = np.zeros((nrPatientsTrain, nrLabels))
T = np.zeros((nrTestRaw, nrFeatures))

#reading in data
X = (pd.read_csv('train_features.csv')).to_numpy()
y = (pd.read_csv('train_labels.csv')).to_numpy()
T = (pd.read_csv('test_features.csv')).to_numpy()

if inDEV:
    print("Input reading complete..")

# standardizing training and test data
X = standardize_data(X, nrTrainingRaw, nrFeatures)
T = standardize_data(T, nrTestRaw, nrFeatures)

if inDEV:
    print("Standardizing data complete..")


# ============================================================================
# Subtask 1
# ============================================================================

if inDEV:
    print("Starting subtask 1..")

# reshape training and test data into new format for all the subtasks
# creating 4 new columns out of every feature column and 12 measurements per patient:
# mean, min, max, trend
X_std = create_feature_matrix(X, nrTrainingRaw, nrFeatures)
T_std = create_feature_matrix(T, nrTestRaw, nrFeatures)

if inDEV:
    print("Reshaping training and test data complete..")

# setup data structure for predicted values
predicted = np.zeros((nrPatientsTest, nrLabels))

# copy first column pid
predicted[:,0] = (np.reshape(T[:,0], (-1,12)))[:,0]

# fitting and predicting for the first 10 columns
for i in range(10):
    
    y_test = y[:,i+1]
    
    """
    # setting up predictor SVM and fitting to given data
    svc1 = svm.SVC(probability=True)
    svc1.fit(X_std, y_test)
    if inDEV:
        print("Fitting " + str(i) + " complete..")
    """
    
    # fitting
    svm1 = svm.SVC(class_weight="balanced")
    svm1.fit(X_std, y_test)
    if inDEV:
        print("Fitting " + str(i) + " complete..")
        
    # predicting
    predicted[:,i+1] = sigmoid_vec(svm1.decision_function(T_std))
    
    if inDEV:
        print("Predicting " + str(i) + " complete..")

# =========================================================================================================
# Subtask 2
# =========================================================================================================

if inDEV:
    print("Starting subtask 2..")

"""
# feature selection with l1 regression via svm
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000).fit(X_std,y_sepsis)
model = SelectFromModel(lsvc, prefit=True)

# after iterating over 1'000'000, the selected features are hardcoded into select_feat
# T_std is reduced to only these selected features before predicting
select_feat = np.array([3,4,5,7,9,11,12,15,16,17,19,20,21,22,25,27,28,30,31,32,34,35])
 

# classification on new training data X_new
X_red = np.zeros((nrPatientsTrain,select_feat.size))
T_red = np.zeros((nrPatientsTest,select_feat.size))

for i in range(select_feat.size):
    X_red[:,i] = X_std[:,select_feat[i]-2]
    T_red[:,i] = T_cond[:,select_feat[i]-2]
    
"""


# classification for sepsis label
y_sepsis = y[:,11]

# fitting ground truth y_sepsis to X_std

svm2 = svm.SVC(class_weight="balanced")
svm2.fit(X_std, y_sepsis)

if inDEV:
    print("Fitting complete..")

# prediction based on fitted svm
# only take second column as it is "probability label is 1"
predicted[:,11] = sigmoid_vec(svm2.decision_function(T_std))

if inDEV:
    print("Prediction complete..")


# =========================================================================================================
# Subtask 3
# =========================================================================================================

if inDEV:
    print("Starting subtask 3..")

# using kernel regression on last 4 columns

for i in range(12, 16, 1):
    # set ground truth as y_test
    y_test = y[:,i]
    
    # setting up svm and fitting to training data
    svm3 = svm.SVR(kernel='rbf')
    svm3.fit(X_std, y_test)
    if inDEV:
        print("Fitting " + str(i) + " complete..")
    
    # predicting for test data
    predicted[:,i] = svm3.predict(T_std)
    if inDEV:
        print("Predicting " + str(i) + " complete..")

# ============================================================================
# POSTPROCESSING
# ============================================================================
data = {'pid': predicted[:,0],
        'LABEL_BaseExcess':predicted[:,1],
        'LABEL_Fibrinogen': predicted[:,2],
        'LABEL_AST': predicted[:,3],
        'LABEL_Alkalinephos': predicted[:,4],
        'LABEL_Bilirubin_total': predicted[:,5],
        'LABEL_Lactate': predicted[:,6],
        'LABEL_TroponinI': predicted[:,7],
        'LABEL_SaO2': predicted[:,8],
        'LABEL_Bilirubin_direct': predicted[:,9],
        'LABEL_EtCO2': predicted[:,10],
        'LABEL_Sepsis': predicted[:,11],
        'LABEL_RRate': predicted[:,12],
        'LABEL_ABPm': predicted[:,13],
        'LABEL_SpO2': predicted[:,14],
        'LABEL_Heartrate': predicted[:,15]}

# creating panda dataframe
df = pd.DataFrame (data, columns = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos',
                                    'LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2',
                                    'LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate',
                                    'LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

# writing dataframe to csv
prediction_dict = dict(method='zip', archive_name='prediction.csv')
df.to_csv('prediction.zip', index=False, float_format='%.3f', compression=prediction_dict)

if inDEV:
    print("Writing output complete..")