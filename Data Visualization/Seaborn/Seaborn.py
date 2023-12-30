# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:36:10 2023

@author: kagan
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data Visualization with Seaborn
# ----------------------------------------------------

# Importing Seaborn and Loading Data
# ----------------------------------

tips = sns.load_dataset('tips')
tips.head()

# Distplot
# --------

# The distplot shows the distribution of a univariate set of observations.
sns.distplot(tips['total_bill'])  # Default distplot
sns.distplot(tips['total_bill'], kde=False)  # Distplot without KDE
sns.distplot(tips['total_bill'], kde=False, bins=30)  # With specified bins

# Jointplot
# ---------

# Jointplot allows us to match up two distplots for bivariate data.
sns.jointplot(x='total_bill', y='tip', data=tips, kind='scatter')
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')

# Pairplot
# --------

# Pairplot will plot pairwise relationships across an entire dataframe.
sns.pairplot(tips)
sns.pairplot(tips, hue='sex', palette='coolwarm')  # With categorical hue

# Rugplot
# -------

# Rugplot - draws a dash mark for every point on a univariate distribution.
sns.rugplot(tips['total_bill'])

# KDEPlot (Kernel Density Estimation Plot)
# ----------------------------------------

# KDE plots replace every single observation with a Gaussian distribution centered around that value.
sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill'])

sns.kdeplot(tips['tip'])
sns.rugplot(tips['tip'])


# Categorical Data Plots with Seaborn
# -----------------------------------

# Loading Dataset
tips = sns.load_dataset('tips')

# Barplot - Show the mean by defult
# -------
# Barplot allows you to aggregate categorical data based on some function (default is mean).

sns.barplot(x='sex', y='total_bill', data=tips)  # Default barplot
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)  # Barplot with standard deviation estimator
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std, hue="smoker", estimator=np.median)  # Adding hue for another categorical dimension

# we can set the argument of "estimator" to be: 
    # Standard Deviation: np.std
    # Variance: np.var
    # Sum: np.sum 

# Countplot
# ---------
# Countplot counts the occurrences.

sns.countplot(x='sex', data=tips)

# Boxplot
# -------
# Boxplot is used to show the distribution of categorical data with quartiles and outliers.

sns.boxplot(x="day", y="total_bill", data=tips, palette='rainbow')  # Basic boxplot
sns.boxplot(x="day", y="total_bill", data=tips, palette='rainbow', hue="smoker")  # Boxplot with hue
sns.boxplot(data=tips, palette='rainbow', orient='h')  # Horizontal boxplot for entire DataFrame
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="coolwarm")  # Boxplot with color palette

# Violinplot
# ----------
# Violinplot shows the distribution of quantitative data across several categorical levels.

sns.violinplot(x="day", y="total_bill", data=tips, palette='rainbow')  # Basic violinplot
sns.violinplot(x="day", y="total_bill", data=tips, hue='sex', palette='Set1')  # Violinplot with hue
sns.violinplot(x="day", y="total_bill", data=tips, hue='sex', split=True, palette='Set1')  # Split violinplot for comparison

# Stripplot and Swarmplot
# -----------------------
# Stripplot and Swarmplot show scatter plots for categorical variables.

sns.stripplot(x="day", y="total_bill", data=tips)  # Basic stripplot
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)  # Stripplot with jitter for density
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, hue='sex', palette='Set1')  # Stripplot with hue and jitter
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, hue='sex', palette='Set1', split=True)  # Split stripplot

sns.swarmplot(x="day", y="total_bill", data=tips)  # Basic swarmplot
sns.swarmplot(x="day", y="total_bill", hue='sex', data=tips, palette="Set1", split=True)  # Swarmplot with hue and split

# Combining Plots
sns.violinplot(x="tip", y="day", data=tips, palette='rainbow')
sns.swarmplot(x="tip", y="day", data=tips, color='black', size=3)

# Factorplot
# ----------
# Factorplot is a versatile plot that can take a 'kind' parameter to adjust the plot type.

sns.factorplot(x='sex', y='total_bill', data=tips, kind='bar')  # Factorplot as bar plot
# The 'kind' parameter can be changed to 'violin' or other plot types for different visualizations.

#######################################################################

# Matrix Plots in Seaborn
# ------------------------

# Loading datasets
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')

# Heatmap
# -------
# A heatmap represents data in a matrix as colors. It's effective for correlation matrices or for any matrix data.

# Correlation matrix for 'tips' dataset
corr = tips.corr()
sns.heatmap(corr)  # Basic heatmap
sns.heatmap(corr, cmap='coolwarm', annot=True)  # Heatmap with color map and annotations

# Creating a pivot table for the 'flights' dataset
pvflights = flights.pivot_table(values='passengers', index='month', columns='year')
sns.heatmap(pvflights)  # Heatmap for 'flights' data
sns.heatmap(pvflights, cmap='magma', linecolor='white', linewidths=1)  # Heatmap with customized style

# Clustermap
# ----------
# The clustermap applies hierarchical clustering and visualizes data as a heatmap with rows and columns clustered.

sns.clustermap(pvflights)  # Basic clustermap for 'flights' data
sns.clustermap(pvflights, cmap='coolwarm', standard_scale=1)  # Normalized clustermap

################################################################

# Grids in Seaborn
# ---------------

# Loading Datasets
iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')

# PairGrid
# --------
# PairGrid allows for plotting pairwise relationships in a dataset.

# Creating a PairGrid
g = sns.PairGrid(iris)
g.map(plt.scatter)  # Mapping a scatter plot to the PairGrid

# Customizing the PairGrid
g = sns.PairGrid(iris)
g.map_diag(plt.hist)  # Histogram on the diagonal
g.map_upper(plt.scatter)  # Scatter plot on the upper triangle
g.map_lower(sns.kdeplot)  # KDE plot on the lower triangle

# Pairplot
# --------
# Pairplot is a simpler version of PairGrid for quick and easy plotting.

sns.pairplot(iris)  # Basic pairplot
sns.pairplot(iris, hue='species', palette='rainbow')  # Pairplot with hue for species

# FacetGrid
# ---------
# FacetGrid is used to create a grid of plots based on features.

# Creating a basic FacetGrid
g = sns.FacetGrid(tips, col="time", row="smoker")
g.map(plt.hist, "total_bill")  # Mapping a histogram to the FacetGrid

# Customizing FacetGrid with hue
g = sns.FacetGrid(tips, col="time", row="smoker", hue='sex')

# Regression Plots in Seaborn
# ----------------------------

# Loading the 'tips' dataset
tips = sns.load_dataset('tips')

# Basic lmplot
# ------------
# lmplot is used to create a scatter plot with a linear fit on top of it.
sns.lmplot(x='total_bill', y='tip', data=tips)

# Enhancing lmplot with categorical data
# --------------------------------------
# The hue parameter adds a color based on a categorical feature.
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex')
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='coolwarm')

# Customizing Markers
# -------------------
# You can customize markers in lmplot by specifying the marker types and scatter_kws for size.
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='coolwarm',
           markers=['o', 'v'], scatter_kws={'s': 100})

# Using a Grid
# ------------
# lmplot allows separation into rows and columns for more detailed categorization.
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')
sns.lmplot(x="total_bill", y="tip", row="sex", col="time", data=tips)
sns.lmplot(x='total_bill', y='tip', data=tips, col='day', hue='sex', palette='coolwarm')

# Adjusting Aspect and Size
# -------------------------
# The aspect and size parameters control the shape and size of the plot.
sns.lmplot(x='total_bill', y='tip', data=tips, col='day', hue='sex', palette='coolwarm',
           aspect=0.6, size=8)
