import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Read dataset and convert to dataframe
data = pd.read_csv('data/Mall_Customers.csv')

# Change columns name
data = data.rename(columns={'Gender': 'gender', 'Age': 'age',
                            'Annual Income (k$)': 'annual_income',
                            'Spending Score (1-100)': 'spending_score'})

# Change categorical data into numerical data
data['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

# Remove CustomerID and gender columns
x = data.drop(['CustomerID', 'gender'], axis=1)

# Create list contained with inertia
clusters = []

for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

# Create inertia plot
fig, ax = plt.subplots(figsize=(8, 4))
sb.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)

ax.set_title('Find Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Create KMeans object
kms = KMeans(n_clusters=5).fit(x)

# Add label column on dataset
x['Labels'] = kms.labels_

# Create KMeans plot with 5 clusters
plt.figure(figsize=(8, 4))
sb.scatterplot(x['annual_income'], x['spending_score'], hue=x['Labels'],
               palette=sb.color_palette('hls', 5))
plt.title('KMeans with 5 clusters')
plt.show()
