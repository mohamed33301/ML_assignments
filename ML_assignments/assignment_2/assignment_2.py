import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        num_sa, num_fe = X.shape
        output = np.ones((num_sa, 1)) 
        for d in range(1, self.degree + 1):
            for i, col_i in enumerate(X.columns): 
                col_v = X[col_i].values
                new_fe = np.power(col_v, d).reshape(-1, 1)
                output = np.hstack((output, new_fe))
                for j, col_j in enumerate(X.columns[i + 1:], start=i + 1):
                    col_values_j = X[col_j].values
                    new_fe = (col_v * col_values_j).reshape(-1, 1)
                    output = np.hstack((output, new_fe))
        return output
    
def Feature_Encoder(X,cols):
    
    lbl = LabelEncoder()
    lbl.fit(X[cols].values)
    X[cols] = lbl.transform(X[cols].values)
    return X
    

data = pd.read_csv("D:\\ass_ml_2_2024\\assignment2dataset.csv")
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

X=data.iloc[:,:5] #Features
print(X)

Y=data['Performance Index'] 

# encoder for " Extracurricular Activities"
cols=('Extracurricular Activities')
X=Feature_Encoder(X,cols)
print(X[cols])
     

#Feature Selection
corr = data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Performance Index'])>0.2]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]

# 80 - 20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

# dynamic degree 
for j in range(2, 3):
    
    model_1_poly_features = PolynomialFeatures(degree=j)
    X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
    poly_model1 = linear_model.LinearRegression()
    poly_model1.fit(X_train_poly_model_1, y_train)
   
    prediction1 = poly_model1.predict(X_train_poly_model_1)
    prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
    
    
    print("\nmodel ", (j - 1),"degree [",j,"] ", "mean_square_error [train] = ", metrics.mean_squared_error(y_train, prediction1))
    print("\t\t\t\t\t  ","mean_square_error [test]  = ", metrics.mean_squared_error(y_test, prediction))
    


