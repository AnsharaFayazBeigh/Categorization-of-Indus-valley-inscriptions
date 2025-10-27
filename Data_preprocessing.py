import pandas as pd
from PIL import Image
import os
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from datetime import datetime
from joblib import dump

# Step 1: Load CSV containing metadata (image names and other details)
print("Loading metadata...")
metadata = pd.read_csv(
    r"D:\Projects\RM_Dataset\folder_contents.csv", 
    header=None, names=['Image Name', 'File Size', 'Creation Time', 'Last Modified Time']
)

# Define the path to your image folder
image_folder_path = r"D:\Projects\RM_Dataset\Indus Dataset"

# Step 2: Process each image and extract HOG features
print("Processing images and extracting HOG features...")
feature_vectors = []  # To collect feature vectors for clustering
processed_image_names = []  # To collect image names that were successfully processed

for index, row in metadata.iterrows():
    image_name = row['Image Name']
    image_path = os.path.join(image_folder_path, image_name)
    
    # Check if the image file exists
    if os.path.exists(image_path):
        print(f"Processing image: {image_name}")
        try:
            # Open and resize the image
            image = Image.open(image_path).resize((128, 128))  # Resize to 128x128
            
            # Convert image to numpy array
            image_array = np.array(image)
            
            # Remove alpha channel if it exists (convert to RGB)
            if image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]
            
            # Convert image to grayscale
            gray_image = color.rgb2gray(image_array)
            
            # Extract HOG features
            fd, _ = hog(gray_image, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            
            # Collect feature vector
            feature_vectors.append(fd)
            processed_image_names.append(image_name)
        
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
    else:
        print(f"Image {image_name} not found.")

# Convert list of feature vectors to numpy array
feature_vectors = np.array(feature_vectors)
print(f"Total images processed: {len(processed_image_names)}")

# Step 3: Apply PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)  # Keep 50 principal components
reduced_features = pca.fit_transform(feature_vectors)

# Save the PCA-transformed features
np.save('reduced_features.npy', reduced_features)
print("PCA-transformed features saved to 'reduced_features.npy'.")

# Save the PCA model
dump(pca, 'pca_model.pkl')
print("PCA model saved to 'pca_model.pkl'.")

# Print explained variance ratio summary
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative explained variance for 50 components: {explained_variance_ratio[-1]:.4f}")

# Step 4: Use Elbow Method to determine optimal k for K-Means clustering
print("Determining optimal k using the Elbow Method...")
k_values = range(1, 11)  # Test for k=1 to k=10
inertia_values = []  # To store inertia values

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_features)
    inertia_values.append(kmeans.inertia_)

# Display the elbow point summary
optimal_k = 4  # Replace with your chosen k value based on previous observations
print(f"Optimal k selected for clustering: {optimal_k}")

# Step 5: Apply K-Means clustering with the selected k
print(f"Applying K-Means clustering with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Save the cluster labels for later use
np.save('cluster_labels.npy', cluster_labels)
print("Cluster labels saved to 'cluster_labels.npy'.")

# Step 6: Evaluate clustering quality
print("Evaluating clustering quality...")
silhouette_avg = silhouette_score(reduced_features, cluster_labels)
calinski_harabasz = calinski_harabasz_score(reduced_features, cluster_labels)
davies_bouldin = davies_bouldin_score(reduced_features, cluster_labels)

# Print all metrics
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Save clustering evaluation metrics
evaluation_metrics = {
    "Silhouette Score": silhouette_avg,
    "Calinski-Harabasz Index": calinski_harabasz,
    "Davies-Bouldin Index": davies_bouldin
}
np.save('clustering_evaluation_metrics.npy', evaluation_metrics)
print("Clustering evaluation metrics saved to 'clustering_evaluation_metrics.npy'.")

# Step 7: Save cluster labels to CSV with versioning
print("Saving cluster labels to CSV...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv_path = f"D:\\Projects\\RM_Dataset\\cluster_labels_{timestamp}.csv"
output_df = pd.DataFrame({
    'Image Name': processed_image_names,
    'Cluster Label': cluster_labels[:len(processed_image_names)]
})
output_df.to_csv(output_csv_path, index=False)
print(f"Cluster labels and image names saved to '{output_csv_path}'.")
