import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

public = pd.read_csv("private_data.csv")

# Read dimensions automatically (exclude 'id' column)
feature_columns = [col for col in public.columns if col != 'id']
X = public[feature_columns].values

# Calculate n and n_clusters
n = X.shape[1]  # number of dimensions/features
n_clusters = 4 * n - 1

print(f"Data dimensions: {n}")
print(f"Number of clusters: {n_clusters}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=80, n_init=200)
labels = kmeans.fit_predict(X_scaled)

submission = pd.DataFrame({'id': public['id'], 'label': labels})
submission.to_csv("private_submission.csv", index=False)

# Example: S2 vs S3
plt.scatter(public['2'], public['3'], c=labels, cmap='tab20', s=5)
plt.xlabel('S2')
plt.ylabel('S3')
plt.title('KMeans Clusters: S2 vs S3')
plt.savefig("kmeans_s2_vs_s3.png", dpi=300)

plt.figure(figsize=(15, 5))

# S1 vs S2
plt.subplot(1, 3, 1)
plt.scatter(public['1'], public['2'], c=labels, cmap='tab20', s=1, alpha=0.5)
plt.xlabel('S1')
plt.ylabel('S2')
plt.title('S1 vs S2')

# S2 vs S3
plt.subplot(1, 3, 2)
plt.scatter(public['2'], public['3'], c=labels, cmap='tab20', s=1, alpha=0.5)
plt.xlabel('S2')
plt.ylabel('S3')
plt.title('S2 vs S3')

# S3 vs S4
plt.subplot(1, 3, 3)
plt.scatter(public['3'], public['4'], c=labels, cmap='tab20', s=1, alpha=0.5)
plt.xlabel('S3')
plt.ylabel('S4')
plt.title('S3 vs S4')

plt.suptitle('Inter-dimensional Relationships in the Private Dataset (Colored by Cluster)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("private_pairwise_scatter_clusters.png", dpi=300)
# plt.show()  # Uncomment if running in an interactive environment
