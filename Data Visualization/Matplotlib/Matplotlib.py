# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:16:13 2023

@author: kagan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Introduction to Matplotlib
# --------------------------

# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
# It provides an object-oriented API for embedding plots into applications.

# Basic Example of Plotting with Matplotlib
# -----------------------------------------

# Creating data for the plots
x = np.linspace(0, 5, 11)
y = x ** 2

# Basic Matplotlib Commands
# -------------------------

# Simple line plot
plt.plot(x, y, 'r')  # 'r' stands for red color
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()

# Creating Multiplots on the Same Canvas
# ---------------------------------------

# Using subplot function to plot multiple plots in one figure
plt.subplot(1, 2, 1)  # (number of rows, number of columns, plot number)
plt.plot(x, y, 'r--')  # 'r--' is a red color with dashed line
plt.subplot(1, 2, 2)
plt.plot(y, x, 'g*-')  # 'g*-' is a green color with star markers
plt.show()

# Matplotlib Object Oriented Method
# ---------------------------------

# Creating a figure instance and then adding axes to it
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
axes.plot(x, y, 'b')
axes.set_xlabel('Set X Label')
axes.set_ylabel('Set Y Label')
axes.set_title('Set Title')

# More complex canvas with two sets of axes
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
# Main plot
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes1')
axes1.set_ylabel('Y_label_axes1')
axes1.set_title('Large Plot')
# Insert plot
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Small Plot')

# Subplots
# --------

# Using the subplots function to create a figure and add axes to it at once
fig, axes = plt.subplots(nrows=1, ncols=2)  # nrows and ncols define the grid of subplots

# Iterating through axes array and plotting
for ax in axes:
    ax.plot(x, y, 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

plt.tight_layout()  # To avoid overlapping of subplots

# Figure Size, Aspect Ratio, and DPI
# ----------------------------------

fig = plt.figure(figsize=(8,4), dpi=100)  # figsize in inches and dpi (dots per inch)

# The same can be applied to subplots
fig, axes = plt.subplots(figsize=(12,3))
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

# Saving Figures
# --------------

fig.savefig("filename.png", dpi=200)  # Saving a figure to a file

# Legends, Labels, and Titles
# ---------------------------

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend(loc=0)  # loc=0 lets matplotlib decide the optimal location

# Setting Colors, Linewidths, Linetypes
# -------------------------------------

fig, ax = plt.subplots()
ax.plot(x, x+1, color="red", linewidth=0.25)
ax.plot(x, x+2, color="red", linewidth=0.50)
ax.plot(x, x+3, color="red", linewidth=1.00, linestyle='-', marker='+')
ax.plot(x, x+4, color="red", linewidth=2.00, linestyle='--', marker='o')

# Special Plot Types
# ------------------

plt.scatter(x, y)  # Scatter plot
plt.hist(x)        # Histogram

# rectangular box plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# rectangular box plot
plt.boxplot(data,vert=True,patch_artist=True);   

##################################################################

# Advanced Matplotlib Concepts
# ----------------------------

# Logarithmic Scale
# -----------------

fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(x, x**2, x, np.exp(x))
axes[0].set_title("Normal scale")
axes[1].plot(x, x**2, x, np.exp(x))
axes[1].set_yscale("log")
axes[1].set_title("Logarithmic scale (y)")

# Custom Tick Labels
# ------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, x**2, x, x**3, lw=2)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)
yticks = [0, 50, 100, 150]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18)

# Scientific Notation
# -------------------

fig, ax = plt.subplots(1, 1)
ax.plot(x, x**2, x, np.exp(x))
ax.set_title("scientific notation")
ax.set_yticks([0, 50, 100, 150])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter)

# Axis Spines
# -----------

fig, ax = plt.subplots(figsize=(6,2))

# Customizing spines
ax.spines['bottom'].set_color('blue')
ax.spines['top'].set_color('blue')
ax.spines['left'].set_color('red')
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_color("none")

# Only ticks on the left side
ax.yaxis.tick_left()

# Plotting some data
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

plt.show()

# Twin Axes
# ---------

fig, ax1 = plt.subplots()
ax1.plot(x, x**2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(x, x**3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")

# Axes where x and y is zero
# --------------------------

fig, ax = plt.subplots()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
xx = np.linspace(-0.75, 1., 100)
ax.plot(xx, xx**3)

# Other 2D Plot Styles
# --------------------

n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1, 4, figsize=(12,3))
axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")
axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")
axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")
axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5)
axes[3].set_title("fill_between")

# Text Annotation
# ---------------

fig, ax = plt.subplots()
ax.plot(xx, xx**2, xx, xx**3)
ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")

# 3D Figures
# ----------

# Create the data for the plots
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))  # Example function

fig = plt.figure(figsize=(14, 6))

# First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# Second subplot with color map
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

plt.show()

