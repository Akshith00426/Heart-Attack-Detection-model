import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the dataset 'heart.csv' loaded as 'data'
data = pd.read_csv("heart.csv")

# Plotting the cholesterol feature distribution
plt.hist(data['chol'], bins=20, color='skyblue', edgecolor='black')
plt.title('Cholesterol Level Distribution')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('Frequency')
plt.show()