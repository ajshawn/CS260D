import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from utils.submodular import faciliy_location_order,get_orders_and_weights

def visualize_facility_location(X, y, coreset_indices, assignments):
    num_clusters = len(coreset_indices)
    colors = plt.cm.get_cmap("tab10", num_clusters)
    a=np.unique(assignments)

    for cluster_id in range(num_clusters):
        cluster_points = X[np.where(assignments==coreset_indices[cluster_id])]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=20,
            color=colors(cluster_id),
            alpha=0.6,
        )

    coreset_points = X[coreset_indices]
    plt.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        s=200,
        color="black",
        marker="x",
        label="Coreset",
    )

    plt.title("Facility Location Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig("FL_test", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    X, y = make_blobs(n_samples=300, centers=5, random_state=42, cluster_std=1.5)
    #X = normalize(X)

    num_per_class = 5
    coreset_indices, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, assignments = get_orders_and_weights(
        num_per_class*5,
        X,
        "euclidean",
        y=y,
        weights=None,
        equal_num=False,
        outdir=".",
        mode="sparse",
        num_n=128,
    )

    visualize_facility_location(X, y, coreset_indices, assignments)