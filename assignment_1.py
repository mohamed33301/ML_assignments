import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


#Loading data
data = pd.read_csv("assignment1dataset.csv")

#understanding the data 
print(data.describe()) 
print(data.shape)
print(data.head())   
print(data.tail())   
print(data.info())   

floatlist=[]  
min_error = float('inf')
best_feature = None
print('----------------------the mean squared error for each model.----------------------------')

for j in data.columns[:-1]:
        X=data[j]
        Y=data['Performance Index']
        X=np.expand_dims(X, axis=1)
        Y=np.expand_dims(Y, axis=1)
        
        L = 0.0001  
        epochs = 1000
        m=0
        c=0
        n = float(len(X)) 
        for i in range(epochs):
            Y_pred = m*X + c  
            D_m = (-2/n) * sum((Y - Y_pred)* X)  
            D_c = (-2/n) * sum(Y - Y_pred)  
            m = m - L * D_m  # Update m
            c = c - L * D_c  # Update c
        prediction = m*X + c
        
        plt.scatter(X, Y)
        plt.xlabel(j, fontsize = 20)
        plt.ylabel('Performance Index', fontsize = 20)
        plt.plot(X, prediction, color='red', linewidth = 3)
        plt.show()
        
        mean_square_error=metrics.mean_squared_error(Y, prediction)
        print('\nMean Square Error of '+j+': ',mean_square_error )
    
        for i in range(5):
            floatlist.insert(i, mean_square_error)
            
        if mean_square_error < min_error:
            min_error = mean_square_error
            best_feature = j 
print('--------------------------------------------------------------------------')
print('\nthe minimum of mean squared error is :  ',min_error)  
print('\nso the best variable (X) that can be used to predict the Performance Index (Y) is :', [ best_feature ] ,'column')
print('--------------------------------------------------------------------------')



