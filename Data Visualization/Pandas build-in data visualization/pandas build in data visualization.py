# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:25:51 2023

@author: kagan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas Built-in Data Visualization
# ----------------------------------

# Setting the style to Seaborn for better aesthetics
sns.set_style('whitegrid')

# Reading in sample data (Replace 'df1' and 'df2' with your data files)
df1 = pd.read_csv('Pandas Built-in Data Visualization/df1', index_col=0)

df2 = pd.read_csv('Pandas Built-in Data Visualization/df2', index_col=0)

# Style Sheets in Matplotlib
# --------------------------
# Using different style sheets for diverse plot aesthetics
plt.style.use('ggplot')
df1['A'].hist()  # Histogram with ggplot style

# Various Plot Types in Pandas
# ----------------------------
# Area plot
df2.plot.area(alpha=0.4)

# Bar plot (stacked)
df2.plot.bar(stacked=True)

# Histogram
df1['A'].plot.hist(bins=50)

# Line plot
df1.plot.line(y='B', figsize=(12,3), lw=1)

# Scatter plot
df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')

# Box plot
df2.plot.box()

