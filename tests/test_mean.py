import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from prediction.prediction import mean_cross_val_scores


class Data:

    def __init__(self):
        col_names = ["id", "clump", "unif_size",
                     "unif_shape", "adhesion", "epi_size", "nuclei",
                     "chromatin", "nucleoli", "mitoses", "class"]
        
        dataset = pd.read_csv("data/raw/breast_cancer.txt", names=col_names, sep=",")
    
        dataset = dataset[(dataset != '?').all(axis=1)]
        dataset['nuclei'] = dataset['nuclei'].astype(int)
        dataset = dataset.drop(columns=["id"])
        dataset['class'] = dataset['class'].replace([2], 0)
        dataset['class'] = dataset['class'].replace([4], 1)
        train_df, test_df = train_test_split(dataset,
                                             test_size=0.3, random_state=123)
        self.X_train = train_df.drop(columns=["class"])
        self.X_test = test_df.drop(columns=["class"])
        self.y_train = train_df["class"]
        self.y_test = test_df["class"]


def test_mean_cross_val_correct_simple():
    """mean_cross_val_scores returns correct mean
    of cross validation with correct types of inputs""" 
    """test if it is a series"""
    """test if num of elements are as expected"""
    """test if elements bigger than 0 and smaller than 1"""
    dataset = Data()
    scale = StandardScaler()
    pipe_knn = make_pipeline(scale,
                             KNeighborsClassifier(n_neighbors=5))
    scoring = ['accuracy']
    result = mean_cross_val_scores(
        pipe_knn, dataset.X_train, dataset.y_train,
        return_train_score=True, scoring=scoring)
    assert all(result) <= 1
    assert all(result) >= 0
    assert result.count() == 4
    assert isinstance(result, pd.core.series.Series)


def test_mean_cross_val_correct():
    """mean_cross_val_scores returns correct mean
    of cross validation with correct types of inputs""" 
    """test if it is a series"""
    """test if num of elements are as expected"""
    """test if elements bigger than 0 and smaller than 1"""
    dataset = Data()
    scale = StandardScaler()
    pipe_knn = make_pipeline(scale,
                             DecisionTreeClassifier(random_state=123))
    scoring = ['accuracy', "f1", "recall", "precision"]
    result = mean_cross_val_scores(
        pipe_knn, dataset.X_train, dataset.y_train,
        return_train_score=True, scoring=scoring)
    assert all(result) <= 1
    assert all(result) >= 0
    assert result.count() == 10
    assert isinstance(result, pd.core.series.Series)
    