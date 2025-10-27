import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style for nicer aesthetics
sns.set(style="whitegrid", palette="muted")

# Read cluster results
cluster_results = pd.read_csv('cluster_labels.csv')

# Ensure correct column names
cluster_results.columns = ['Image Name', 'Cluster']

# Visualize cluster distribution with improved aesthetics
plt.figure(figsize=(10, 6))
sns.countplot(data=cluster_results, x='Cluster', palette='coolwarm', edgecolor='black')
plt.title('Cluster Distribution', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(rotation=0)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


features = np.random.rand(100, 10)  # Example: 100 images, 10 features per image

# Correlation heatmap for extracted features
plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(features.T)  # Assuming 'features' is a 2D array of extracted features
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

print("Visualization completed.")
