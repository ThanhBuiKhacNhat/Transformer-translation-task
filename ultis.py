import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('metrics.csv')

# Extract the data for the two columns
column1 = data['test_loss']
column2 = data['test_accuracy']

# Create the plot
plt.plot(column1, label='test_loss')
plt.plot(column2, label='test_accuracy')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('')
plt.xticks(range(len(column1)), range(len(column1)))
plt.title('Plot of Loss and Accuracy')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Save the plot
plt.savefig('Results\plot.png')