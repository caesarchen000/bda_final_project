import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

public = pd.read_csv("public_data.csv")

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
submission.to_csv("public_submission.csv", index=False)
