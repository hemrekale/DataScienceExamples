# [source] : https://medium.com/@kapil.mathur1987/matplotlib-an-introduction-to-its-object-oriented-interface-a318b1530aed

import matplotlib.pyplot as plt
fig = plt.figure()

# Generate a grid of 2x2 subplots and get
# axes object for 1st location
ax1 = fig.add_subplot(2,2,1)
ax1.set_title('First Location')

# Get the axes object for subplot at 2nd 
# location
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('Second Location')

# Get the axes object for subplot at 3rd 

ax3 = fig.add_subplot(2,2,3)
ax3.set_xlabel('Third Location')

# Get the axes object for subplot at 4th 
# location
ax4 = fig.add_subplot(2,2,4)
ax4.set_xlabel('Fourth Location')

# Display
plt.show()

# Generate data for plots 
x = [1, 2, 3, 4, 5]
y = x

# Get an empty figure
fig1 = plt.figure()

# Get the axes instance at 1st location in 1x1 grid
ax = fig1.add_subplot(1,1,1)

# Generate the plot
ax.plot(x, y)

# Set labels for x and y axis
ax.set_xlabel('X--->')
ax.set_ylabel('Y--->')

# Set title for the plot
ax.set_title('Simple XY plot')

# Display the figure
plt.show()

# Two plots in same figure
# Function to get the square of each element in the list
def list_square(a_list):
    return [element**2 for element in a_list]

# Multiple plot in same subplot window
# plot y = x and z = x^2 in the same subplot window
fig2 = plt.figure()

x = [1, 2, 3, 4, 5]
y = x
z = list_square(x)

# Get the axes instance
ax = fig2.add_subplot(1,1,1)

# Plot y vs x as well as z vs x. label will be used by ax.legend() method to generate a legend automatically
ax.plot(x, y, label='y')
ax.plot(x, z, label='z')
ax.set_xlabel("X------>")

# Generate legend
ax.legend()

# Set title
ax.set_title('Two plots one axes')

# Display
plt.show()

# Function to get the square of each element in the list
def list_square(a_list):
    return [element**2 for element in a_list]

# Multiple subplots in same figure
fig3 = plt.figure()
x = [1, 2, 3, 4, 5]
y = x
z = list_square(x)

# Divide the figure into 1 row 2 column grid and get the
# axes object for the first column
ax1 = fig3.add_subplot(1,2,1)

# plot y = x on axes instance 1
ax1.plot(x, y)

# set x and y axis labels
ax1.set_xlabel('X------>')
ax1.set_ylabel('Y------>')
ax1.set_title('y=x plot')

# Get second axes instance in the second column of the 1x2 grid
ax2 = fig3.add_subplot(1,2,2)

# plot z = x^2
ax2.plot(x, z)
ax2.set_xlabel('X---------->')
ax2.set_ylabel('z=X^2--------->')
ax2.set_title('z=x^2 plot')

# Generate the title for the Figure. Note that this is different then the title for individual plots
plt.suptitle("Two plots in a figure")
plt.show()
