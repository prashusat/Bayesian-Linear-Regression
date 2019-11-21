import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import time
#fucntion to read data in pandas dataframe


def read_data(data_file,regression_values_file):
    df_data=pd.read_csv(data_file, sep=',',header=None)
    df_regression_values=pd.read_csv(regression_values_file, sep=',',header=None)
    return df_data,df_regression_values
##df_data,df_regression_values=read_data("train-100-10.csv","trainR-100-10.csv")
##npy_data,npy_labels=convert_df_to_numpy(df_data,df_regression_values)
##w=learn_w(150,10,npy_data,npy_labels)
##predicted=predict_label(w,npy_data)
##mse=mse_calculator(npy_labels,predicted)
#function to convert pandas dataframe to numpy arrays
def convert_df_to_numpy(dataframe_data,dataframe_values):
    return dataframe_data.values,dataframe_values.values


def function_to_split(df_data,df_regression_values):
    number_of_rows=df_data.shape[0]
    number_of_rows_per_split=int(math.floor(number_of_rows/10))
    data_splits=[]
    label_splits=[]
    j=0
    for i in range(0,10):
        temp_df_data=df_data[j:j+number_of_rows_per_split]
        temp_df_label=df_regression_values[j:j+number_of_rows_per_split]
        j=j+number_of_rows_per_split
        data_splits+=[temp_df_data]
        label_splits+=[temp_df_label]
    return data_splits,label_splits
        
    
##df_data,df_regression_values=read_data("train-100-10.csv","trainR-100-10.csv")
##data_splits,label_splits=function_to_split(df_data,df_regression_values)

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
def fold_10_mse_vs_lambda_plotter(avg_mse_error):
    lambda_values=[i for i in range(0,150)]
    plt.title("10-fold MSE vs lambda_values") 
    plt.xlabel("lambda") 
    plt.ylabel("Mean-squared error") 
    plt.plot(lambda_values,avg_mse_error,color='olive')
    plt.show()

def task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values):
    start_time=time.time()
    mse_values=[]
    for lambda_value in range(0,150):
        avg_mse=0
        for i in range(0,10):
            
            test_npy_data,test_npy_labels=convert_df_to_numpy(data_splits[i],label_splits[i])
            vertical_stack_data =pd.concat([data_splits[j] for j in range(0,10) if i!=j], axis=0)
            vertical_stack_label=pd.concat([label_splits[j] for j in range(0,10) if i!=j], axis=0)
            #converting dataframes to numpy arrays
            train_npy_data,train_npy_labels=convert_df_to_numpy(vertical_stack_data,vertical_stack_label)
                    
            i_matrix=identity_matrix_generator(train_npy_data.shape[1])
            w=learn_w(lambda_value,i_matrix,train_npy_data,train_npy_labels)
            #predicted labels
            test_predicted=predict_label(w,test_npy_data)
            mse_test_temp=mse_calculator(test_npy_labels,test_predicted)
            avg_mse+=mse_test_temp
            
        mse_values+=[avg_mse/10]
    print("The best value of lambda would be:",mse_values.index(min(mse_values)))
    
    
    #converting dataframes to numpy arrays
    train_npy_data,train_npy_labels=convert_df_to_numpy(train_df_data,train_df_regression_values)
    test_npy_data,test_npy_labels=convert_df_to_numpy(test_df_data,test_df_regression_values)
    i_matrix=identity_matrix_generator(train_npy_data.shape[1])
    w=learn_w(mse_values.index(min(mse_values)),i_matrix,train_npy_data,train_npy_labels)
    test_predicted=predict_label(w,test_npy_data)
    mse_test_temp=mse_calculator(test_npy_labels,test_predicted)
    print("Mean squared error on test set:",mse_test_temp)
    print("Time elapsed",time.time()-start_time)
    fold_10_mse_vs_lambda_plotter(mse_values) 
    
    
                 
while(1):
    print("\n","1. 100-10","\n","2. 100-100","\n","3. 1000-100","\n","4. crime","\n","5. wine","\n","6. Challenge Dataset","\n")
    i=int(input("Choose the number corresponding to the dataset you want to perform task-3.1 on:"))
    if i==1:
        train_df_data,train_df_regression_values=read_data("train-100-10.csv","trainR-100-10.csv")
        test_df_data,test_df_regression_values=read_data("test-100-10.csv","testR-100-10.csv")
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
        
    elif i==2:
        train_df_data,train_df_regression_values=read_data("train-100-100.csv","trainR-100-100.csv")
        test_df_data,test_df_regression_values=read_data("test-100-100.csv","testR-100-100.csv")
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
    elif i==3:
        train_df_data,train_df_regression_values=read_data("train-1000-100.csv","trainR-1000-100.csv")
        test_df_data,test_df_regression_values=read_data("test-1000-100.csv","testR-1000-100.csv")
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
    elif i==4:
        train_df_data,train_df_regression_values=read_data("train-crime.csv","trainR-crime.csv")
        test_df_data,test_df_regression_values=read_data("test-crime.csv","testR-crime.csv")
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
    elif i==5:
        train_df_data,train_df_regression_values=read_data("train-wine.csv","trainR-wine.csv")
        test_df_data,test_df_regression_values=read_data("test-wine.csv","testR-wine.csv")
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
    elif i==6:
        train=input("Enter train dataset filename:")
        trainR=input("Enter train regression values dataset filename:")
        test=input("Enter test dataset filename:")
        testR=input("Enter test regression values dataset filename:")
        train_df_data,train_df_regression_values=read_data(train,trainR)
        test_df_data,test_df_regression_values=read_data(test,testR)
        data_splits,label_splits=function_to_split(train_df_data,train_df_regression_values)
        task_3_1(data_splits,label_splits,train_df_data,train_df_regression_values,test_df_data,test_df_regression_values)
        
    else:
        print("Invalid Input")



    




















