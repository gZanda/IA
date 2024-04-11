import matplotlib.pyplot as plt
import numpy as np

# Generate initial data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create initial plot
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dynamic Plot Example')

# Make changes to the plot
# For example, let's update the color and add a legend
plt.plot(x, np.cos(x), color='red', label='cos(x)')
plt.legend()

# Update the plot
plt.draw()
plt.pause(2)  # pause for a short time to allow the plot to be displayed

# More changes can be made here...
# For example, let's change the line style
plt.plot(x, np.tan(x), color='green', linestyle='dashed', label='tan(x)')
plt.legend()

# Update the plot again
plt.draw()
plt.pause(2)

# Keep the plot displayed until the user closes it
plt.show()
