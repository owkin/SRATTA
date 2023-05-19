def check_result(clustered_centers, list_truth_centers):
    """
    Check that the constructed clusters are all correct

    """
    for cluster in clustered_centers:
        if len(set([list_truth_centers[c] for c in cluster])) > 1:
            print(f"Error with cluster {cluster}")
            print(f"List center {[list_truth_centers[c] for c in cluster]}")
            raise ValueError
