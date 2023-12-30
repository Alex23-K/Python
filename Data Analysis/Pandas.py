# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:00:48 2022

@author: kagan
"""

import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(101)

# Pandas Series
# -----------------------------

# Demonstrating Pandas Series with Simple Custom Labels

# Creating Series from a list with default and custom indices
my_list = [10, 20, 30]

# The series with a default index
series_from_list = pd.Series(data=my_list)  
# 0    10
# 1    20
# 2    30
# dtype: int64 

# Defining simple custom labels for the Series
labels = ['a', 'b', 'c']

# Creating a Pandas Series with the list and custom labels
series_with_labels = pd.Series(data=my_list, index=labels)  
series_with_labels 
# a    10
# b    20
# c    30
# dtype: int64 

# Accessing elements using custom labels
element_a = series_with_labels['a']  # Accessing the element with label 'a'
element_a # output:  10


# Creating Series from a NumPy array
arr = np.array([10, 20, 30])
series_from_array = pd.Series(arr)  # 
# 0    10
# 1    20
# 2    30
# dtype: int32

 # Custom index
array_series_labels = pd.Series(arr, labels) 
# a    10
# b    20
# c    30
# dtype: int32


# Creating Series from a dictionary
d = {'a': 10, 'b': 20, 'c': 30}
series_from_dict = pd.Series(d)  # Index will be from dictionary keys

# Series hold various data types
series_with_strings = pd.Series(data=labels)
series_with_functions = pd.Series([sum, print, len])  # Holding functions


# Indexing in Pandas Series
# -------------------------

# Creating two series with different indices
ser1 = pd.Series([1, 2, 3, 4], index=['USA', 'Ukraine', 'Israel', 'Japan'])
ser1 # output: 
#  USA       1
# Ukraine    2
# Israel     3
# Japan      4
# dtype: int64

ser2 = pd.Series([1, 2, 5, 4], index=['USA', 'Germany', 'Italy', 'Japan'])

# Accessing data from series using index
data_from_ser1 = ser1['USA']  # Accessing data from ser1, output: 1

# Operations with Series
# ----------------------

# Adding two series together - Indices are aligned automatically
ser1_plus_ser2 = ser1 + ser2  # NaN for non-matching indices
ser1_plus_ser2 # output: 
# Germany    NaN
# Israel     NaN
# Italy      NaN
# Japan      8.0
# USA        2.0
# Ukraine    NaN
# dtype: float64

###############################################################


# Creating a DataFrame
# --------------------

# Creating a DataFrame from a random numpy array with custom row and column labels
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df
#           W         X         Y         Z
# A  2.706850  0.628133  0.907969  0.503826
# B  0.651118 -0.319318 -0.848077  0.605965
# C -2.018168  0.740122  0.528813 -0.589001
# D  0.188695 -0.758872 -0.933237  0.955057
# E  0.190794  1.978757  2.605967  0.683509

# DataFrame Selection and Indexing
# --------------------------------

# Selecting a single column w - returns a Series
column_w = df['W']
column_w
# A    2.706850
# B    0.651118
# C   -2.018168
# D    0.188695
# E    0.190794
# Name: W, dtype: float64


# Selecting multiple columns - returns a DataFrame
columns_w_z = df[['W', 'Z']]

# DataFrame column types
type(df['W'])  # pandas.core.series.Series
type(df)  # pandas.core.frame.DataFrame

# Creating a new column 'new' (whuch is the sum of columns w+y)
df['new'] = df['W'] + df['Y']
print(round(df,1)) #output: 

#     W    X    Y    Z  new
# A  2.7  0.6  0.9  0.5  3.6
# B  0.7 -0.3 -0.8  0.6 -0.2
# C -2.0  0.7  0.5 -0.6 -1.5
# D  0.2 -0.8 -0.9  1.0 -0.7
# E  0.2  2.0  2.6  0.7  2.8


# Removing Columns
# Use inplace=True to modify the DataFrame in place
df.drop('new', axis=1)  # axis = 1 - specify to columns,  axis = 0 is defult - specified for the rows.
 #  inplace=True - to remove permenantly the "new" column withut casting. 

df
# drop rows: 
df.drop('E',axis=0) 
 
# Selecting Rows
# Using label-based indexing (loc) and integer-based indexing (iloc)
df.loc['C']
df.iloc[2]
# output:
# W     -2.018168
# X      0.740122
# Y      0.528813
# Z     -0.589001
# new   -1.489355
# Name: C, dtype: float64

# Selecting subsets of rows and columns
df.loc['B', 'Y'] #  -0.848

# multiple selection by rows and columns
df.loc[['A', 'B'], ['W', 'Y']]
# Output: 
#           W         Y
# A  2.706850  0.907969
# B  0.651118 -0.848077

# Conditional Selection
# Applying conditions to DataFrames
condition = df > 3
#       W      X      Y      Z    new
# A  False  False  False  False   True
# B  False  False  False  False  False
# C  False  False  False  False  False
# D  False  False  False  False  False
# E  False  False  False  False  False

# Present the numeric values by the condition
df_by_condition = df[condition]
print(df_by_condition)
# output:
#    W   X   Y   Z       new
# A NaN NaN NaN NaN  3.614819
# B NaN NaN NaN NaN       NaN
# C NaN NaN NaN NaN       NaN
# D NaN NaN NaN NaN       NaN
# E NaN NaN NaN NaN       NaN

# Presenting the rows where the condition by column W was met.  
df_positive_w = df[df['W'] > 0.5]

# Combining conditions
df_combined_conditions = df[(df['W'] > 0) & (df['Y'] > 1)]

# More on Indexing
# -----------------

# Resetting index
df_reset = df.reset_index()  # Adds an 'index' column and makes the current index numerical

# Setting a new index
new_index = 'CA NY WY OR CO'.split()
df['States'] = new_index 
df.set_index('States', inplace=True) # define the 'states' column as index column

print(round(df,1))
 #         W    X    Y    Z  new
# States                         
# CA      2.7  0.6  0.9  0.5  3.6
# NY      0.7 -0.3 -0.8  0.6 -0.2
# WY     -2.0  0.7  0.5 -0.6 -1.5
# OR      0.2 -0.8 -0.9  1.0 -0.7
# CO      0.2  2.0  2.6  0.7  2.8


# Multi-Index and Index Hierarchy
# -------------------------------

# Creating a Multi-Indexed DataFrame
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1, 2, 3, 1, 2, 3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df_multi = pd.DataFrame(randn(6,2), index=hier_index, columns=['A', 'B'])
df_multi.index.names = ['Group', 'Num']

print(df_multi)
# output: 
#                  A         B
# Group Num                    
# G1    1   -1.005187 -0.741790
#       2    0.187125 -0.732845
#       3   -1.382920  1.482495
# G2    1    0.961458 -2.141212
#       2    0.992573  1.192241
#       3   -1.046780  1.292765


# Accessing data in Multi-Indexed DataFrame
group_g1 = df_multi.loc['G1']

group_g1 # output:
#            A         B
# Num                    
# 1   -1.005187 -0.741790
# 2    0.187125 -0.732845
# 3   -1.382920  1.482495

group_g1_num1 = df_multi.loc['G1'].loc[1]

group_g1_num1 # output: 

# A   -1.005187
# B   -0.741790

# Cross-section using xs
df_xs = df_multi.xs(1, level='Num')

print(df_xs)
 #              A         B
# Group                    
# G1    -1.00519    -0.74179
# G2     0.961458   -2.14121

##############################################################

# Handling Missing Data
# ---------------------

# Creating a DataFrame with missing values
df = pd.DataFrame({'A':[1, 2, np.nan],
                   'B':[5, np.nan, np.nan],
                   'C':[1, 2, 3]})

print (df) #output:
#     A    B  C
# 0  1.0  5.0  1
# 1  2.0  NaN  2
# 2  NaN  NaN  3    

# Dropping rows with missing values
df_dropna_rows = df.dropna()  # By default, dropna() drops rows with any missing values
print(df_dropna_rows) # output:
    
#     A    B  C
# 0  1.0  5.0  1


# Dropping columns with missing values
df_dropna_cols = df.dropna(axis=1)  # Dropping columns with missing values
df_dropna_cols
#    C
# 0  1
# 1  2
# 2  3


# Dropping rows with a certain number of non-NA values (threshold)
df_dropna_thresh = df.dropna(thresh=2)  # Keeps rows with at least 2 non-NA values
#     A    B  C
# 0  1.0  5.0  1
# 1  2.0  NaN  2


# Filling missing values
df_fillna = df.fillna(value=10)  # Replaces NA with a specified value
print(df_fillna) # Output: 
#     A     B  C
# 0   1.0   5.0  1
# 1   2.0  10.0  2
# 2  10.0  10.0  3


df_fillna_mean = df['A'].fillna(value=df['A'].mean())  # Filling NA in column 'A' with the mean of 'A'
print(df_fillna_mean) # Output:    
# 0    1.0
# 1    2.0
# 2    1.5
# Name: A, dtype: float64


# GroupBy Operations
# ------------------

# Creating a DataFrame for GroupBy examples
data = {'Company':['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
        'Person':['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
        'Sales':[200, 120, 340, 124, 243, 350]}
df_group = pd.DataFrame(data)
print(df_group) # Output: 
 #   Company   Person  Sales
 # 0    GOOG      Sam    200
 # 1    GOOG  Charlie    120
 # 2    MSFT      Amy    340
 # 3    MSFT  Vanessa    124
 # 4      FB     Carl    243
 # 5      FB    Sarah    350


# Grouping data by 'Company' column
by_comp = df_group.groupby("Company")
print(by_comp )

# Aggregate methods after GroupBy method
mean_sales = by_comp.mean()  # Mean sales by company
# output: 
 #         Sales
 # Company       
 # FB       296.5
 # GOOG     160.0
 # MSFT     232.0

sum_sales = by_comp.sum()  # Total sales by company
std_sales = by_comp.std()  # Standard deviation of sales by company
min_sales = by_comp.min()  # Minimum sales by company
max_sales = by_comp.max()  # Maximum sales by company
count_sales = by_comp.count()  # Counting entries by company

print(count_sales)
#          Person  Sales
# Company               
# FB            2      2
# GOOG          2      2
# MSFT          2      2

# Describe function after GroupBy
comp_desc = by_comp.describe()  # Descriptive statistics by company
comp_desc_transpose = by_comp.describe().transpose()  # Transpose for better readability

# Accessing specific company data
fb_data = by_comp.sum().loc['FB']  # Sum of sales for 'FB'
# Sales    593
# Name: FB, dtype: int64

# Descriptive statistics for 'GOOG'
goog_data = by_comp.describe().transpose()['GOOG']  

###################################################################

# Example DataFrames for Demonstrating Merging, Joining, and Concatenating
# -----------------------------------------------------------------------

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

# Concatenation
# -------------

# Concatenating DataFrames along rows (axis=0 is default)
concatenated_df = pd.concat([df1, df2, df3])
print(concatenated_df) # Output: 
'''   A    B    C    D
0    A0   B0   C0   D0
1    A1   B1   C1   D1
2    A2   B2   C2   D2
3    A3   B3   C3   D3
4    A4   B4   C4   D4
5    A5   B5   C5   D5
6    A6   B6   C6   D6
7    A7   B7   C7   D7
8    A8   B8   C8   D8
9    A9   B9   C9   D9
10  A10  B10  C10  D10
11  A11  B11  C11  D11    
'''  
        
# Concatenating DataFrames along columns (axis=1)
concatenated_df_cols = pd.concat([df1, df2, df3], axis=1)
concatenated_df_cols

'''   A    B    C    D    A    B    C    D    A    B    C    D
0    A0   B0   C0   D0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
1    A1   B1   C1   D1  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
2    A2   B2   C2   D2  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
3    A3   B3   C3   D3  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
4   NaN  NaN  NaN  NaN   A4   B4   C4   D4  NaN  NaN  NaN  NaN
5   NaN  NaN  NaN  NaN   A5   B5   C5   D5  NaN  NaN  NaN  NaN
6   NaN  NaN  NaN  NaN   A6   B6   C6   D6  NaN  NaN  NaN  NaN
7   NaN  NaN  NaN  NaN   A7   B7   C7   D7  NaN  NaN  NaN  NaN
8   NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   A8   B8   C8   D8
9   NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   A9   B9   C9   D9
10  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  A10  B10  C10  D10
11  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  A11  B11  C11  D11
'''

# Merging
# -------

# Example DataFrames for merging
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

print(left) # output:
'''  key   A   B
0  K0  A0  B0
1  K1  A1  B1
2  K2  A2  B2
3  K3  A3  B3
''' 

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

print(right) # output:
    '''key   C   D
    0  K0  C0  D0
    1  K1  C1  D1
    2  K2  C2  D2
    3  K3  C3  D3
    '''

# Merging on a common key
merged_df = pd.merge(left, right, how='inner', on='key')
print(merged_df)
''' output: 
     key   A   B   C   D
  0  K0  A0  B0  C0  D0
  1  K1  A1  B1  C1  D1
  2  K2  A2  B2  C2  D2
  3  K3  A3  B3  C3  D3
  '''


# More complicated example of merging
left_complicated = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                                 'key2': ['K0', 'K1', 'K0', 'K1'],
                                 'A': ['A0', 'A1', 'A2', 'A3'],
                                 'B': ['B0', 'B1', 'B2', 'B3']})
right_complicated = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                                  'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})

merged_complicated = pd.merge(left_complicated, right_complicated, on=['key1', 'key2'])

print(merged_complicated)
'''
   key1 key2   A   B   C   D
0   K0   K0  A0  B0  C0  D0
1   K1   K0  A2  B2  C1  D1
2   K1   K0  A2  B2  C2  D2
'''

# Joining
# -------

# Example DataFrames for joining
left_join = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                          'B': ['B0', 'B1', 'B2']},
                         index=['K0', 'K1', 'K2']) 

print(left_join) # Output: 
'''  A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2
''' 


right_join = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                           'D': ['D0', 'D2', 'D3']},
                          index=['K0', 'K2', 'K3'])

print(right_join) # Output:
'''     C   D
   K0  C0  D0
   K2  C2  D2
   K3  C3  D3
''' 

# Joining DataFrames
joined_df = left_join.join(right_join)
print(joined_df) # Output: 
''' 
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
''' 

# Outer join
joined_outer = left_join.join(right_join, how='outer')
'''
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3
''' 
#########################################################################

# Data Input and Output using Pandas
# ----------------------------------

# CSV Files
# ----------------------------------
pwd # get the working dictory 

# Reading from a CSV file
df_csv = pd.read_csv('example.csv')  # path to the example file

# Writing to a CSV file
df_csv.to_csv('output.csv', index=False)  # Saving DataFrame to a CSV, excluding the index


# HTML (HyperText Markup Language) Files
# --------------------------------------

# Reading HTML content
# Assuming the webpage at the URL contains tables
html_data = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list')
df_html = html_data[0]  # Assuming the first table is what we want

df_html.head(3) # output: 
#               Bank NameBank       CityCity  ... Closing DateClosing  FundFund
# 0             Citizens Bank       Sac City  ...    November 3, 2023     10545
# 1  Heartland Tri-State Bank        Elkhart  ...       July 28, 2023     10544
# 2       First Republic Bank  San Francisco  ...         May 1, 2023     10543


