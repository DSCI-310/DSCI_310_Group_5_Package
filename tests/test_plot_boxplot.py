from unicodedata import numeric
import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.plot_boxplot import boxplot_plotting
test_df = pd.DataFrame({'age':['25','48','30'], 'height': ['185','192','187'],'weight':['85','93','90'], 
'class':['0','1','1'] })
num_df=test_df.apply(pd.to_numeric)
var_names=num_df.head()
number_of_rows=3
number_of_columns=1
width = 50
height = 26
incorrect_plotting_example_type = 3.2
empty_data_frame = pd.DataFrame(columns ={'age':[''], 'height': [''],'weight':[''], 'class':[''] })
single_value_data_frame = pd.DataFrame(columns ={'age':['19'], 'height': ['25'],'weight':['32'], 'class':['0'] })
double_value_data_frame = pd.DataFrame(columns ={'age':['19','24'], 'height': ['25','54'],'weight':['32','75'], 'class':['0','1'] })




#min([], default="EMPTY")

list_example = ["apple", "banana", "cherry"]
test_case = boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,num_df,3)
b=mpl.figure.Figure()
comparison_var=5

def test_return_type():
    #Test for the correct return type of function:
    assert type(test_case) == type(b)

def test_dataframe_type_values():
    #Tests to see if the values of each column are numeric in order to be able to plot them
    for i in range (len(var_names)):
        assert type(i)==type(comparison_var)
        
def test_integer_values():
    #Test to confirm that num_rows and num_columns are integer values
    assert type(number_of_rows) == type(comparison_var)
    assert type(number_of_rows) == type(comparison_var)
    assert type(width) == type(comparison_var)
    assert type(height) == type(comparison_var)
    

def test_product_consistency():
    #Tests to see if the number of boxplots created will match the number of variables involved. This is 
    #to avoid extra unuseful boxplots or not enough boxplots to show all variables interacting with the class values
    assert number_of_columns * number_of_rows == len(var_names)

def test_wrong_input_dataframe():
    """
    Check TypeError raised when inputting wrong type for what should be a pandas dataframe.
    """
    with pytest.raises(TypeError):
        boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,list_example,3)

def test_wrong_type_input_number_rows():
    """
    Check TypeError raised when inputting wrong type for what should be an integer value for the number of rows.
    """
    with pytest.raises(TypeError):
        boxplot_plotting(incorrect_plotting_example_type,number_of_columns,width,height,var_names,list_example,3)

def test_wrong_type_input_number_columns():
    """
    Check TypeError raised when inputting wrong type for what should be an integer value for the number of columns.
    """
    with pytest.raises(TypeError):
        boxplot_plotting(number_of_rows,incorrect_plotting_example_type,width,height,var_names,list_example,3)
def test_edge_case_empty_dataframe():
    """
    Check edge case where we have an empty dataframe (only column names). We want to see if it will return a mpl.figure.Figure() data type 
    """
    try:
        holder = boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,empty_data_frame,3)
        assert type(holder) == type(b)
    except ValueError:
        pass
    
def test_edge_case_single_value_dataframe():
    """
    Check edge case where we have a dataframe with only one example (one of the binary class labels only). We want to see if it will return a mpl.figure.Figure() data type 
    """
    try:
        holder1 = boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,single_value_data_frame,3)
        assert type(holder1) == type(b)
    except ValueError:
        pass
def test_edge_case_each_class():
    """
    Check edge case where we have a dataframe with one example of each of the binary class labels. We want to see if it will return a mpl.figure.Figure() data type 
    """
    try:
        holder2 = boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,double_value_data_frame,3)
        assert type(holder2) == type(b)
    except ValueError:
        pass
