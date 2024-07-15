import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = 'Mall_Customers.csv'  
df = pd.read_csv(file_path)

# Print the first few rows to understand the structure
print(df.head())

# Selecting the relevant columns for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Applying K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)  # Adjusting the number of clusters as needed
df['Cluster'] = kmeans.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of customers')
plt.show()

