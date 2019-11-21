import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#fucntion to read data in pandas dataframe


def read_data(data_file,regression_values_file):
    df_data=pd.read_csv(data_file, sep=',',header=None)
    df_regression_values=pd.read_csv(regression_values_file, sep=',',header=None)
    return df_data,df_regression_values


#function to fetch the shape of data and the labels
def get_shape(dataframe_data,dataframe_values):
    return dataframe_data.shape,dataframe_values.shape

#function to convert pandas dataframe to numpy arrays
def convert_df_to_numpy(dataframe_data,dataframe_values):
    return dataframe_data.values,dataframe_values.values



#generates identity matrix based on the number of features present in the data
def identity_matrix_generator(number_of_features):
    return np.identity(number_of_features)

#fucntion to learn w
def learn_w(lambda_value,identity_matrix,data_in_npy_format,labels_in_npy_format):
    product_of_lambda_identity=np.multiply(lambda_value,identity_matrix)
    product_covariance=np.dot(np.transpose(data_in_npy_format),data_in_npy_format)
    sum_of_lambda_identity_product_covariance=np.add(product_of_lambda_identity,product_covariance)
    inverse_of_sum=np.linalg.inv(sum_of_lambda_identity_product_covariance)
    product_of_data_and_label=np.dot(np.transpose(data_in_npy_format),labels_in_npy_format)
    product_of_inverse_and_phi_T_t=np.dot(inverse_of_sum,product_of_data_and_label)
    return product_of_inverse_and_phi_T_t
    



#fucntion to multiply w with data in order to predict labels
def predict_label(w,data_in_npy_format):
    predicted_labels=np.dot(data_in_npy_format,w)
    return predicted_labels

#fucntion used to calculate mean squared error
def mse_calculator(actual_labels,predicted_labels):
    difference_in_prediction=np.subtract(predicted_labels,actual_labels)
    squared_difference_in_prediction=np.square(difference_in_prediction)
    sum_squared_difference_in_prediction=np.sum(squared_difference_in_prediction)
    return (sum_squared_difference_in_prediction/actual_labels.shape[0])

##df_data,df_regression_values=read_data("train-100-10.csv","trainR-100-10.csv")
##npy_data,npy_labels=convert_df_to_numpy(df_data,df_regression_values)
##w=learn_w(150,10,npy_data,npy_labels)
##predicted=predict_label(w,npy_data)
##mse=mse_calculator(npy_labels,predicted)
def mse_vs_lambda_plotter(train_mse,test_mse,lambda_values):
    plt.title("MSE vs lambda") 
    plt.xlabel("lambda") 
    plt.ylabel("Mean-squared error") 
    plt.plot(lambda_values,train_mse,color='olive',label='train-mse')
    plt.plot(lambda_values,test_mse,color='blue',label="test-mse")
    plt.legend()
    plt.show()

def task_1(train_data_file,train_regression_values_file,test_data_file,test_regression_values_file):
    #reading train and test data into dataframes
    train_df_data,train_df_regression_values=read_data(train_data_file,train_regression_values_file)
    test_df_data,test_df_regression_values=read_data(test_data_file,test_regression_values_file)
    #converting dataframes to numpy arrays
    train_npy_data,train_npy_labels=convert_df_to_numpy(train_df_data,train_df_regression_values)
    test_npy_data,test_npy_labels=convert_df_to_numpy(test_df_data,test_df_regression_values)
    mse_for_train=[]
    mse_for_test=[]
    lambda_values=[]
    i_matrix=identity_matrix_generator(train_npy_data.shape[1])
    for lambda_value in range(1,150):
        #learning w

        lambda_values.append(lambda_value)
        w=learn_w(lambda_value,i_matrix,train_npy_data,train_npy_labels)
        #predicted labels
        train_predicted=predict_label(w,train_npy_data)
        test_predicted=predict_label(w,test_npy_data)
        mse_train_temp=mse_calculator(train_npy_labels,train_predicted)
        mse_test_temp=mse_calculator(test_npy_labels,test_predicted)
        mse_for_train.append(mse_train_temp)
        mse_for_test.append(mse_test_temp)
    

        
    mse_vs_lambda_plotter(mse_for_train,mse_for_test,lambda_values)
while(1):
    print("\n","1. 100-10","\n","2. 100-100","\n","3. 1000-100","\n","4. crime","\n","5. wine","\n","6. challenge dataset","\n")
    i=int(input("Choose the number corresponding to the dataset you want to perform task-1 on:"))
    if i==1:
        task_1("train-100-10.csv","trainR-100-10.csv","test-100-10.csv","testR-100-10.csv")
    elif i==2:
        task_1("train-100-100.csv","trainR-100-100.csv","test-100-100.csv","testR-100-100.csv")
    elif i==3:
        task_1("train-1000-100.csv","trainR-1000-100.csv","test-1000-100.csv","testR-1000-100.csv")
    elif i==4:
        task_1("train-crime.csv","trainR-crime.csv","test-crime.csv","testR-crime.csv")
    elif i==5:
        task_1("train-wine.csv","trainR-wine.csv","test-wine.csv","testR-wine.csv")
    elif i==6:
        train=input("Enter train dataset filename:")
        trainR=input("Enter train regression values dataset filename:")
        test=input("Enter test dataset filename:")
        testR=input("Enter test regression values dataset filename:")
        task_1(train,trainR,test,testR)
    else:
        print("INVALID OPTION")

    
        
        
        
        









    






        
        
        
    
    


    
