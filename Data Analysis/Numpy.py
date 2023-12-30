# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:43:39 2022

@author: kagan
"""

import numpy as np


# Basic Array Creation and Operations
# -----------------------------------

# Creating a 1D array
arr = np.arange(0, 11)  # Output: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Creating a 2D array from a list of lists
my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
two_d_np_array = np.array(my_matrix)  
# Output: array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic arithmetic operations on arrays
arr_add = arr + arr  # Output: array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
arr_mul = arr * arr  # Output: array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
arr_sub = arr - arr  # Output: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Advanced Array Operations
# --------------------------

# Array Broadcasting
arr[:5] = 100
# Output: array([100, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10])

# 2D Array Operations and Indexing
arr_2d = np.array(([5, 10, 15], [20, 25, 30], [35, 40, 45]))

# Slicing top right corner 2x2 from 2D array
arr_2d_slice = arr_2d[:2, 1:]  # Output: array([[10, 15], [25, 30]])

# Fancy Indexing
arr2d = np.zeros((10, 10))
for i in range(arr2d.shape[1]):
    arr2d[i] = i
fancy_indexed = arr2d[[2, 4, 6, 8]]  
# Output: array([[2., 2., 2. ... 2., 2., 2.] ... [8., 8., 8., ... 8.. 8., 8.]])

# Boolean Indexing and Conditional Selection
arr_boolean = arr[arr > 4]  
arr_boolean # Output: array([5, 6, 7, 8, 9, 10])

# Generating Arrays with Specific Patterns
# ----------------------------------------

# Generating arrays of zeros and ones
zeros_array = np.zeros(3)
zeros_array # Output: array([0., 0., 0.])
ones_array = np.ones(4) # Output: array([1., 1., 1., 1.])

# Using linspace to create evenly spaced values
lin_space = np.linspace(1, 10, 7)  
lin_space # Output: array([ 1. ,  2.5,  4. ,  5.5,  7. ,  8.5, 10. ])

# Creating an identity matrix
identity_matrix = np.eye(5)  
identity_matrix # Output: array([[1., 0., 0., 0., 0.],
                              # [0., 1., 0., 0., 0.],
                              # [0., 0., 1., 0., 0.],
                              # [0., 0., 0., 1., 0.],
                              # [0., 0., 0., 0., 1.]])
    

# Random Number Generation
# ------------------------

# Uniform distribution
rand_array = np.random.rand(5)  # Output: 5 random numbers from a uniform distribution over [0, 1)

# Normal distribution
randn_array = np.random.randn(5)  # Output: 5 samples from the "standard normal" distribution

# Random integers
rand_int_array = np.random.randint(1, 100, 10)  # Output: Array of 10 random integers between 1 and 100


# Basic arithmetic operations on arrays
arr_add = arr + arr  # Output: [ 0  2  4  6  8 10 12 14 16 18 20]
arr_mul = arr * arr  # Output: [  0   1   4   9  16  25  36  49  64  81 100]
arr_sub = arr - arr  # Output: [0 0 0 0 0 0 0 0 0 0 0]

# Division by array (note: division by zero will result in nan)
arr_div = arr / arr  # Output: [nan  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

# Division by scalar (note: division by zero will result in inf)
div_by_scalar = 1 / arr  # Output: [       inf 1.         0.5        0.33333333 0.25       0.2        ...]

# Power operation on array
arr_power = arr ** 3  # Output: [   0    1    8   27   64  125  216  343  512  729 1000]

# Subtracting a scalar from an array
arr_minus_100 = arr - 100  # Output: [-100  -99  -98  -97  -96  -95  -94  -93  -92  -91  -90]


# Universal Array Functions and Attributes
# ----------------------------------------
# Restarting the array
arr = np.arange(0, 11) 

# Applying universal functions
sqrt_arr = np.sqrt(arr)  # Taking square roots of the array elements
exp_arr = np.exp(arr)    # Calculating the exponential of all elements in the array
sin_arr = np.sin(arr)    # Applying sine function to all elements in the array
log_arr = np.log(arr)    # Applying logarithm function to all elements in the array

# Finding max, min and their index positions
max_value = arr.max()  # Maximum value in arr: output: 10 
max_index = arr.argmax()  # Index of the maximum value in arr
min_value = arr.min()  # Minimum value in arr
min_index = arr.argmin()  # Index of the minimum value in arr

# Array shape and data type
arr_shape = arr.shape  # Shape of arr
arr_dtype = arr.dtype  # Data type of arr elements

# Practical Examples and Exercises
# --------------------------------

# List and String Manipulation
s = 'Hi there David!'
split_list = s.split()  # Output: ['Hi', 'there', 'David!']

# Nested List and Dictionary Indexing
lst = [1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7]

# Grab hello from the list
hello_from_list = lst[3][1][2][0]  # Output: 'hello'

# Create a dictionary
d = {'k1':[1, 2, 3, {'tricky':['oh', 'man', 'inception', {'target':[1, 2, 3, 'hello']}]}]}

# Grab the word hello from a dictonery
hello_from_dict = d['k1'][3]['tricky'][3]['target'][3]  # Output: 'hello'

# Using lambda and filter to select words
seq = ['soup', 'dog', 'salad', 'cat', 'great']
filtered_seq = list(filter(lambda word: word[0] == 's', seq))  
filtered_seq # Output: ['soup', 'salad']
