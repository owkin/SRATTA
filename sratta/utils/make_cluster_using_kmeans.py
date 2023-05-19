import numpy as np
from sklearn.cluster import KMeans


def clusters_samples_with_kmeans(list_recovered_samples, config):
    if len(list_recovered_samples) < config.parameters.num_centers:
        return [set(sample) for sample in list_recovered_samples]
    list_clusters = []
    kmeans = KMeans(
        n_clusters=config.parameters.num_centers,
        random_state=0,
    )
    clusters_kmeans = kmeans.fit(np.array(list_recovered_samples)).labels_
    for center in range(config.parameters.num_centers):
        list_clusters.append(set(np.argwhere(clusters_kmeans == center).ravel()))

    return list_clusters
