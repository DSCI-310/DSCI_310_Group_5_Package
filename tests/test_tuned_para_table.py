import numpy as np
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.tuned_para_table import tuned_para_table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
from dsci_prediction.dsci_prediction import *

y_vals = np.random.choice([0, 1], size=40)
df = pd.DataFrame({'x1': np.linspace(2,10,40), 'x2': np.linspace(4,10,40),
                   'x3': np.linspace(4,10,40), 'class': y_vals})
train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)
X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
search = GridSearchCV(pipe_knn,
                      param_grid={'kneighborsclassifier__n_neighbors': range(1, 10),
                                  'kneighborsclassifier__weights': ['uniform', 'distance']},
                      cv=10, 
                      n_jobs=-1,  
                      scoring="recall", 
                      return_train_score=True)


def test_tuned_para_table_knn():
    """
    Test table readability (columns name) and return type
    """
    table = tuned_para_table(search, X_train, y_train) 
    assert list(table.columns)[0] == 'kneighborsclassifier__n_neighbors'
    assert list(table.columns)[1] == 'kneighborsclassifier__weights'
    assert list(table.columns)[2] == 'best_score'
    assert isinstance(table, pd.core.frame.DataFrame)


def test_edge_case_four_obs_2_splits():
    """
    Test table readability (columns name) and return type  when there are four observations
    in X_train and y_train with respect to cv = 2 in search
    """
    X_min = X_train[:4]
    y_min = pd.DataFrame({'class':[1,0,0,1]})
    search = GridSearchCV(pipe_knn,
                          param_grid={'kneighborsclassifier__n_neighbors': [1, 2],
                                  'kneighborsclassifier__weights': ['uniform', 'distance']},
                          cv=2,
                          n_jobs=-1, 
                          scoring="recall", 
                          return_train_score=True)
    table = tuned_para_table(search, X_min, y_min) 
    assert list(table.columns)[0] == 'kneighborsclassifier__n_neighbors'
    assert list(table.columns)[1] == 'kneighborsclassifier__weights'
    assert list(table.columns)[2] == 'best_score'
    assert isinstance(table, pd.core.frame.DataFrame)


def test_wrong_input_X_train():
    """
    Check TypeError raised when inputting wrong type of X_train.
    """
    with pytest.raises(TypeError):
        tuned_para_table(search, "wrong input type", y_train)


def test_wrong_input_search():
    """ 
    Check TypeError raised when inputting wrong type of search.
    """
    with pytest.raises(TypeError):
        tuned_para_table("wrong input type", X_train, y_train)