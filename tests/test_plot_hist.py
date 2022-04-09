import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.plot_hist import plot_hist_overlay
import pytest

# simpler dataframe
df0 = pd.DataFrame(np.linspace(2,10,20), columns=["x1"])
df1 = pd.DataFrame(np.linspace(4,10,20), columns=["x1"])

# more complex (edge_cases) dataframe
square_dict0 = {"feat1" : np.linspace(2,10,20),
               "feat2" : np.logspace(2,10,20),
               "feat3" : np.linspace(2,10,20),
               "feat4" : np.linspace(2,10,20)}

square_dict1 = {"feat1" : np.linspace(4,10,20),
               "feat2" : np.logspace(4,10,20),
               "feat3" : np.linspace(4,10,20),
               "feat4" : np.linspace(4,10,20)}

sqr_df0 = pd.DataFrame(square_dict0)
sqr_df1 = pd.DataFrame(square_dict1)
feats = ["feat1", "feat2", "feat3", "feat4"]

dict0 = {"feat1" : np.linspace(2,10,20),
        "feat2" : np.logspace(2,10,20),
        "feat3" : np.linspace(2,10,20)}
    
dict1 = {"feat1" : np.linspace(4,10,20),
        "feat2" : np.logspace(4,10,20),
        "feat3" : np.linspace(4,10,20)}

notsqr_df0 = pd.DataFrame(dict0)
notsqr_df1 = pd.DataFrame(dict1)
feats2 = ["feat1", "feat2", "feat3"]

# initial objects to test with
labels = ["0 - negative","1 - positive"]
fig, axe = plot_hist_overlay(df0, df1, ["x1"], labels=labels)
fig_test, ax_test = plt.subplots()

def test_return_type():
    """
    Test for the correct return type of the function is an Axes object
    """
    assert type(axe) == type(ax_test)
    assert type(fig) == type(fig_test)

def test_incorrect_parameter_type():
    """
    Test if a TypeError is raise if any input parameter does not have the correct type
    as required in the function documentation
    """
    with pytest.raises(TypeError):
        plot_hist_overlay("df0", df1, labels=labels, columns=feats)

    with pytest.raises(TypeError):
        plot_hist_overlay(df0, "df1", labels=labels, columns=feats)
    
    with pytest.raises(TypeError):
        plot_hist_overlay("df0", "df1", labels="label", columns=feats)
    
    with pytest.raises(TypeError):
        plot_hist_overlay(df0, df1, labels=labels, columns="col")
    
    with pytest.raises(TypeError):
        plot_hist_overlay("df0", df1, labels=labels, columns=["x1"],fig_no=1) 

def test_genearal_readability():
    """
    Test for the correct label for X,y axis, legend and title
    """
    assert axe.get_xlabel() == "X1"
    assert axe.get_ylabel() == "Count"
    assert axe.get_legend().get_texts()[0].get_text() == labels[0]
    assert axe.get_legend().get_texts()[1].get_text() == labels[1]
    
    # default figure number fig_no = 1
    assert axe.get_title() == "Figure 1.1: Histogram of X1 for each target class label"
    
    # figure number supplied by fig_no
    fig2,axe2 = plot_hist_overlay(df0, df1, ["x1"], fig_no="2", labels=labels, ec="white")
    
    assert axe2.get_title() == "Figure 2.1: Histogram of X1 for each target class label"
    plt.cla()

def test_edge_case_1(): 
    """
    Test to see if the function correctly plot for dataframe with only 1 column
    """
    fig3,axe3 = plot_hist_overlay(df0, df1, ["x1"], fig_no="2", labels=labels, ec="white")
    
    assert type(axe3) == type(ax_test)
    assert axe3.get_legend().get_texts()[0].get_text() == labels[0]
    assert axe3.get_legend().get_texts()[1].get_text() == labels[1]
    assert axe3.get_title() == "Figure 2.1: Histogram of X1 for each target class label"

def test_edge_case_square():
    """
    Test functionality when the whole figure is a square of subplots, e.g: dimension of a square (2,2)
    with the number of features is perfect squares e.g: 4 features can be arrange into a 2x2 square figure
    """
    sqr_fig, sqr_axe = plot_hist_overlay(sqr_df0, sqr_df1, columns=feats, labels=labels, fig_no="3")
    
    # basic functionalities
    assert type(sqr_axe) == type(ax_test)
    assert sqr_axe.get_xlabel() == "Feat4"
    assert sqr_axe.get_ylabel() == "Count"
    assert sqr_axe.get_legend().get_texts()[0].get_text() == labels[0]
    assert sqr_axe.get_legend().get_texts()[1].get_text() == labels[1]

    # test correct dimension that is automatically calculated by our function
    sqr_fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    assert sqr_axe.get_gridspec().ncols == ax4.get_gridspec().ncols
    assert sqr_axe.get_gridspec().ncols == 2
    assert sqr_axe.get_gridspec().nrows == ax4.get_gridspec().nrows
    assert sqr_axe.get_gridspec().nrows == 2
    assert sqr_axe.get_title() == "Figure 3.4: Histogram of Feat4 for each target class label"

def test_edge_case_not_square():
    """
    Test functionality when the figure is a square but the number of features are not quite perfect square
    Will also test if we can render an empty subplot
    """
    notsqr_fig, notsqr_axe = plot_hist_overlay(notsqr_df0, notsqr_df1, columns=feats2, labels=labels, fig_no="4")

    # basic functionalities
    assert type(notsqr_axe) == type(ax_test)
    assert notsqr_axe.get_xlabel() == "Feat3"
    assert notsqr_axe.get_ylabel() == "Count"
    assert notsqr_axe.get_legend().get_texts()[0].get_text() == labels[0]
    assert notsqr_axe.get_legend().get_texts()[1].get_text() == labels[1]

    # test correct dimension that is automatically calculated by our function
    # in this case 3 features will also be plotted in a square grid 2x2, but the last subplot is just empty
    sqr_fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    assert notsqr_axe.get_gridspec().ncols == ax3.get_gridspec().ncols
    assert notsqr_axe.get_gridspec().ncols == 2
    assert notsqr_axe.get_gridspec().nrows == ax3.get_gridspec().nrows
    assert notsqr_axe.get_gridspec().nrows == 2
    assert notsqr_axe.get_title() == "Figure 4.3: Histogram of Feat3 for each target class label" 
