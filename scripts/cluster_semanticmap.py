

import glob
import pandas as pd
# copy xbd dataset to OEM
import shutil
import os
import rasterio
import torch
import tqdm
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from tqdm   import tqdm
def  kmeans_cluseter(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    # 获取聚类结果
    clusters = kmeans.labels_
    return kmeans, clusters


def  extract_features(mask, num_classes):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    features = []
    for i in range(num_classes):
        pixel_count = np.sum(mask == i)
        features.append(pixel_count)
    features = np.array(features) / np.prod(mask.shape)  # 归一化
    return features

def  save_results( lbl_paths, clusters, num_clusters, output_base_dir ):
    results = {'path': lbl_paths, 'cluster': clusters}
    results_df = pd.DataFrame(results)

    # 保存到CSV文件中
    results_df.to_csv(os.path.join(output_base_dir, f"clustering_result_{num_clusters}.csv"), index=False)

    for cluster in range(num_clusters):
        cluster_dir = os.path.join(output_base_dir, f'cluster_{cluster}')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    for idx, row in results_df.iterrows():
        src_path = row['path'].replace("labels", "images")
        cluster = row['cluster']
        dst_dir = os.path.join(output_base_dir, f'cluster_{cluster}')
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

def visual_plot(num_clusters, features, clusters):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    plt.figure(figsize=(10, 7))
    for cluster in range(num_clusters):
        cluster_indices = np.where(clusters == cluster)
        plt.scatter(features_pca[cluster_indices, 0], features_pca[cluster_indices, 1], label=f'Cluster {cluster}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Features and K-means Clusters')
    plt.legend()
    plt.show()

import pickle

def save_kmeans_model(kmeans, out_dir, num_clusters):
    model_path = os.path.join(out_dir, f'kmeans_model_{num_clusters}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)

def load_kmeans_model(model_path):
    if not os.path.exists(model_path):
        return  None
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans

def main(args):
    data_dir = args.data_dir#   r"F:\data\OpenEarthMap\Size_256\train"
    lbl_paths = glob.glob(os.path.join(data_dir,  "*.tif"))

    num_original_classes = args.num_classes
    num_clusters = args.num_clusters

    features_list = []
    for lbl_path in tqdm(lbl_paths):
        with rasterio.open(lbl_path) as src:
            mask = src.read(1)  # 读取第一个波段
            features = extract_features(mask, num_original_classes )
            features_list.append(features)
    features = np.array(features_list)

    # elbow_method_with_knee_locator(features)
    # silhouette_method(features)
    kmeans, clusters = kmeans_cluseter(features, num_clusters)

    if args.plot:
        visual_plot(num_clusters, features, clusters)


    out_dir = args.output_dir

    save_results(lbl_paths, clusters, num_clusters, out_dir)
    save_kmeans_model(kmeans, out_dir,num_clusters)
    print("Done.")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_elbow_point(inertia):
    x = range(1, len(inertia) + 1)
    kn = KneeLocator(x, inertia, curve='convex', direction='decreasing')
    return kn.knee


# 使用KneeLocator来找到肘部点
from kneed import KneeLocator


def elbow_method_with_knee_locator(features, mink = 1, maxk = 20):
    inertia = []
    K = range(mink, maxk + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')

    elbow_point = find_elbow_point(inertia)
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    plt.show()

    print(f"The optimal number of clusters is: {elbow_point}")


def silhouette_method(features):
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()




if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, default=r"F:\data\OpenEarthMap\Size_256\train\images")
    parser.add_argument("--output_dir", type=str, required=False, default=r"F:\data\OpenEarthMap\Size_256\train\classfications")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    main(args)