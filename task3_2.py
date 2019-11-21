import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import random
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

def find_lambda(beta,data_matrix):
    return np.multiply(beta,np.dot(np.transpose(data_matrix),data_matrix))

def find_Sn(alpha,identity_matrix,beta,data_matrix):
    product=find_lambda(beta,data_matrix)
    return np.linalg.inv(np.add(np.multiply(alpha,identity_matrix),product))

def find_Mn(beta,Sn,data_matrix,label_matrix):
    Sn_x_data_matrix=np.dot(Sn,np.transpose(data_matrix))
    above_x_t=np.dot(Sn_x_data_matrix,label_matrix)
    return np.multiply(beta,above_x_t)
def find_gamma(beta,data_matrix,alpha,Sn):

    lambda_matrix=find_lambda(beta,data_matrix)
    lamb=np.linalg.eigvals(lambda_matrix)
    gamma_final=0
    for i in lamb:
        gamma_final+=i/(alpha+i)
    return gamma_final
def find_alpha(gamma,Mn):
    alpha_final=gamma/np.dot(np.transpose(Mn),Mn)
    return alpha_final[0][0]
def find_beta(data_matrix,label_matrix,Mn,gamma):
    phi_x_Mn=np.dot(data_matrix,Mn)
    sigma_term=np.sum(np.square(np.subtract(label_matrix,phi_x_Mn)))
    beta=sigma_term/(data_matrix.shape[0]-gamma)
    return 1/beta

def predict_label(w,data_in_npy_format):
    predicted_labels=np.dot(data_in_npy_format,w)
    return predicted_labels

#fucntion used to calculate mean squared error
def mse_calculator(actual_labels,predicted_labels):
    difference_in_prediction=np.subtract(predicted_labels,actual_labels)
    squared_difference_in_prediction=np.square(difference_in_prediction)
    sum_squared_difference_in_prediction=np.sum(squared_difference_in_prediction)
    return (sum_squared_difference_in_prediction/actual_labels.shape[0])

def task3_2(data_matrix,label_matrix,test_df_data,test_df_regression_values):
    #randomly initialising alpha and beta
    alpha=random.randrange(1,10)
    beta=random.randrange(1,10)
    #print(alpha)
    test_npy_data,test_npy_labels=convert_df_to_numpy(test_df_data,test_df_regression_values)
    alpha_array=[]
    beta_array=[]
    errors_array=[]
    for i in range(0,100):
        lambda_value=find_lambda(beta,data_matrix)
        
        Sn=find_Sn(alpha,np.identity(data_matrix.shape[1]),beta,data_matrix)
        Mn=find_Mn(beta,Sn,data_matrix,label_matrix)
        gamma=find_gamma(beta,data_matrix,alpha,Sn)
        alpha=find_alpha(gamma,Mn)
        beta=find_beta(data_matrix,label_matrix,Mn,gamma)
        test_predicted=predict_label(Mn,test_npy_data)
        error_on_test_set=mse_calculator(test_npy_labels,test_predicted)
        alpha_array.append(alpha)
        beta_array.append(beta)
        errors_array.append(error_on_test_set)
        if (i>=1):
            if (alpha_array[i]-alpha_array[i-1])<0.0001 and (beta_array[i]-beta_array[i-1])<0.0001:
                print("Coverged in ",i," iterations.")
                print("Alpha value has converged to:",alpha)
                print("Beta value has converged to:",beta)
                print("Lambda value has converged to:",alpha/beta)
                print("Mean Squared Error on test set:",errors_array[i])
                
                
                
                break
        



    


while(1):
    print("\n","1. 100-10","\n","2. 100-100","\n","3. 1000-100","\n","4. crime","\n","5. wine","\n","6. Challenge Dataset","\n")
    i=int(input("Choose the number corresponding to the dataset you want to perform task-1 on:"))
    if i==1:
        df_data,df_regression_values=read_data("train-100-10.csv","trainR-100-10.csv")
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data("test-100-10.csv","testR-100-10.csv")
    elif i==2:
        df_data,df_regression_values=read_data("train-100-100.csv","trainR-100-100.csv")
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data("test-100-100.csv","testR-100-100.csv")
    elif i==3:
        df_data,df_regression_values=read_data("train-1000-100.csv","trainR-1000-100.csv")
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data("test-1000-100.csv","testR-1000-100.csv")
    elif i==4:
        df_data,df_regression_values=read_data("train-crime.csv","trainR-crime.csv")
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data("test-crime.csv","testR-crime.csv")
    elif i==5:
        df_data,df_regression_values=read_data("train-wine.csv","trainR-wine.csv")
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data("test-wine.csv","testR-wine.csv")
    elif i==6:
        train=input("Enter train dataset filename:")
        trainR=input("Enter train regression values dataset filename:")
        test=input("Enter test dataset filename:")
        testR=input("Enter test regression values dataset filename:")
        df_data,df_regression_values=read_data(train,trainR)
        data_matrix,label_matrix=convert_df_to_numpy(df_data,df_regression_values)
        test_df_data,test_df_regression_values=read_data(test,testR)

    start_time=time.time()
    error_on_test_set=task3_2(data_matrix,label_matrix,test_df_data,test_df_regression_values)
    end_time=time.time()
    print("Time elapsed: ",end_time-start_time)


    
