import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prediction.prediction import plot_hist_overlay

df0 = pd.DataFrame(np.linspace(2,10,20), columns=["x1"])
df1 = pd.DataFrame(np.linspace(4,10,20), columns=["x1"])
labels = ["0 - negative","1 - positive"]
plot = plot_hist_overlay(df0, df1, "x1", labels=labels)
plot2 = plot_hist_overlay(df0, df1, "x1", fig_no="1.0", labels=labels, ec="white")
fig, ax = plt.subplots()

def test_return_type():
    """
    Test for the correct return type of the function is an Axes object
    """
    assert type(plot) == type(ax)

def test_readability():
    """
    Test for the correct label for X,y axis, legend and title
    """
    assert plot.get_xlabel() == "X1"
    assert plot.get_ylabel() == "Count"
    assert plot.get_legend().get_texts()[0].get_text() == labels[0]
    assert plot.get_legend().get_texts()[1].get_text() == labels[1]
    # default figure number fig_no = 1
    assert plot.get_title() == "Figure 1: Histogram of X1 for each target class label"
    # figure number supplied by fig_no
    assert plot2.get_title() == "Figure 1.0: Histogram of X1 for each target class label"