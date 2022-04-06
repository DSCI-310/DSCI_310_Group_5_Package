from unicodedata import numeric
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from prediction.prediction import boxplot_plotting

test_df = pd.DataFrame({'age':['25','48','30'], 'height': ['185','192','187'],'weight':['85','93','90'], 
'class':['0','1','1'] })

num_df=test_df.apply(pd.to_numeric)
var_names=num_df.head()
number_of_rows=3
number_of_columns=1
test_case = boxplot_plotting(number_of_rows,number_of_columns,0.5,0.5,var_names,num_df,3)
b=mpl.text.Text()
comparison_var=5

def test_return_type():
    #Test for the correct return type of function:
    assert type(test_case) == type(b)

def test_dataframe_type_values():
    #Tests to see if the values of each column are numeric in order to be able to plot them
    for i in range (len(var_names)):
        assert type(i)==type(comparison_var)

def test_product_consistency():
    #Tests to see if the number of boxplots created will match the number of variables involved. This is 
    #to avoid extra unuseful boxplots or not enough boxplots to show all variables interacting with the class values
    assert number_of_columns * number_of_rows == len(var_names)