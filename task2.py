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

def sample_data(df_data,df_regression_values,number_of_samples):
    sampled_df_index=df_data.sample(n=number_of_samples).index.values
    return df_data.ix[sampled_df_index],df_regression_values.ix[sampled_df_index]
##df_data,df_regression_values=read_data("train-1000-100.csv","trainR-1000-100.csv")
##    
##df_data_sampled,df_regression_values_sampled=sample_data(df_data,df_regression_values,20)
##print(df_data_sampled.shape,df_regression_values_sampled.shape)


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
def mse_vs_size_plotter(avg_mse_error_1,avg_mse_error_2,avg_mse_error_3,size):
    plt.title("MSE vs Size of Data") 
    plt.xlabel("Size of dataset") 
    plt.ylabel("Mean-squared error") 
    plt.plot(size,avg_mse_error_1,color='grey',label="lambda=5")
    plt.plot(size,avg_mse_error_2,color='red',label="lambda=90")
    plt.plot(size,avg_mse_error_3,color='blue',label="lambda=145")
    plt.legend()
    plt.show()
def task_2(lambda_value):
    avg_mse_error=[]
    df_data,df_regression_values=read_data("train-1000-100.csv","trainR-1000-100.csv")
    for k in range(len(m)):
        avg_mse_error.append(0)
    index_to_update=0       
    for i in m:
        
        for j in range(0,10):
            train_df_data,train_df_regression_values=sample_data(df_data,df_regression_values,i)
            test_df_data,test_df_regression_values=read_data("test-1000-100.csv","testR-1000-100.csv")
            #converting dataframes to numpy arrays
            train_npy_data,train_npy_labels=convert_df_to_numpy(train_df_data,train_df_regression_values)
            test_npy_data,test_npy_labels=convert_df_to_numpy(test_df_data,test_df_regression_values)
            
            i_matrix=identity_matrix_generator(train_npy_data.shape[1])
            
            w=learn_w(lambda_value,i_matrix,train_npy_data,train_npy_labels)
            #predicted labels
            test_predicted=predict_label(w,test_npy_data)
            mse_test_temp=mse_calculator(test_npy_labels,test_predicted)
            avg_mse_error[index_to_update]+=mse_test_temp
            
        index_to_update+=1
    avg_mse_error[:]=[x / 10 for x in avg_mse_error]
    return avg_mse_error



m=[i for i in range(50,950,100)]
avg_mse_error_1=task_2(5)
avg_mse_error_2=task_2(90)
avg_mse_error_3=task_2(145)
mse_vs_size_plotter(avg_mse_error_1,avg_mse_error_2,avg_mse_error_3,m)










            
        
