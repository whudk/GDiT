from random import random

import cv2
import torch

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import json
import random
import seaborn as sns
# 创建图数据列表


def visualize_graph(g):


    G = to_networkx(g, to_undirected=True)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=15)
    plt.title(f'Graph ')
    plt.show()


def apply_color_palette_torch(image):
    color_palette = {
        "0-1": [0, 0, 0],
        "1-2": [128, 0, 0],
        "2-3": [0, 255, 36],
        "3-4": [148, 148, 148],
        "4-5": [255, 255, 255],
        "5-6": [34, 97, 38],
        "6-7": [0, 69, 255],
        "7-8": [75, 181, 73],
        "8-9": [222, 31, 7],
    }

    # Convert the grayscale image to a tensor
    b, c, h, w = image.shape
    gray_tensor = image[:,0,:,:]#torch.tensor(gray_image, dtype=torch.int32)

    # Initialize the RGB tensor
    rgb_tensor = torch.zeros((b, h, w , 3), dtype=torch.uint8).to(image.device)

    # Process each color range
    for key, color in color_palette.items():
        start, end = map(int, key.split('-'))
        mask = (gray_tensor >= start) & (gray_tensor < end)
        # Repeat the color across the channel dimension and apply the mask
        rgb_tensor[mask,:] = torch.tensor(color, dtype=torch.uint8).to(image.device)

    return rgb_tensor.permute(0,3,1,2)
def replace_nodes_with_words(graph_data, captions):
    node_mapping = captions["Level-1"]
    new_x = []
    for node in graph_data.x:
        word = node_mapping[str(int(node.item()))]["words"]
        new_x.append(word)
    return new_x
def replace_adj_with_words(adj_list, node_mapping):
    new_adj_list = np.array(adj_list).flatten()
    vals = np.unique(new_adj_list)
    new_nodes = []
    for val in vals:
        new_node  = node_mapping[str(val)]["words"]
        new_nodes.append(new_node)
    return new_nodes


def show_graph_from_node_edges(node_path, edge_path, graph_save_path=None, heatmap_save_path=None):
    # Load node and edge data
    node = np.load(node_path)
    edge = np.load(edge_path)

    # Create graph data
    graph_data = Data(x=torch.tensor(node, dtype=torch.float), edge_index=torch.tensor(edge, dtype=torch.long))
    G = to_networkx(graph_data, to_undirected=True)

    # Create a figure for the graph
    plt.figure(figsize=(10, 8))

    # Visualize the graph
    pos = nx.spring_layout(G)
    edge_colors = ['blue' if i % 2 == 0 else 'darkblue' for i in range(edge.shape[1])]

    nx.draw(G, pos, node_color='skyblue', edge_color=edge_colors, node_size=500, font_size=6)

    # Set node labels
    labels = {i: str(i) for i in range(len(node))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Add captions at the bottom
    plt.figtext(0.5, 0.01, "Graph Visualization of Nodes and Edges", wrap=True, horizontalalignment='center', fontsize=8)

    # Save the plot as a PNG file
    if graph_save_path:
        plt.savefig(graph_save_path, format='png', bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Display the edge matrix
    edge_matrix = np.zeros((len(node), len(node)))
    for i, j in edge.T:
        edge_matrix[i, j] = 1

    print("Edge Matrix:")
    print(edge_matrix)

    # Visualize the edge matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(edge_matrix, cmap='Blues', annot=False, cbar=False, square=True,
                xticklabels=[], yticklabels=[])

    # Save the heatmap as a PNG file
    if heatmap_save_path:
        plt.savefig(heatmap_save_path, format='png', bbox_inches='tight')

    plt.show()


def show_graph(x_cond, instance_map, edges, save_paths):
    OEM_dict = {
        0: 'Background',
        1: 'Bareland',
        2: 'Rangeland',
        3: 'Developed space',
        4: 'Road',
        5: 'Tree',
        6: 'Water',
        7: 'Agriculture land',
        8: 'Building'
    }
    color_palette = {
        0: [0, 0, 0],
        1: [128, 0, 0],
        2: [0, 255, 36],
        3: [148, 148, 148],
        4: [255, 255, 255],
        5: [34, 97, 38],
        6: [0, 69, 255],
        7: [75, 181, 73],
        8: [222, 31, 7],
    }

    # Assuming x_cond is [Batch, Height, Width] and instance_map is [Batch, Height, Width]
    x_cond_np = x_cond.cpu().squeeze(0).numpy()  # Convert x_cond to numpy and remove batch dimension
    instance_map_np = instance_map.cpu().squeeze(0).numpy()  # Convert instance_map to numpy and remove batch dimension
    # Ensure instance_map_np is an integer array

    # Visualize the first instance from each


    # Plot x_cond
    plt.figure(figsize=(5, 5))
    plt.imshow(x_cond_np, cmap="viridis")  # Display x_cond with a suitable color map
    plt.title("x_cond")
    plt.colorbar(label="Value Range (0-9)")
    plt.axis("off")
    plt.savefig(save_paths["x_cond"], bbox_inches='tight', dpi=100)
    plt.close()


    # Plot instance_map
    plt.figure(figsize=(5, 5))
    plt.imshow(instance_map_np, cmap="tab20")  # Assuming instance_map is categorical
    plt.title("instance_map")
    plt.colorbar(label="Instance IDs")
    plt.axis("off")
    plt.savefig(save_paths["instance_map"], bbox_inches='tight', dpi=100)
    plt.close()


    # Plot graph on instance_map
    plt.figure(figsize=(5, 5))
    plt.imshow(instance_map_np, cmap="tab20")  # Use the instance map as background
    plt.title("Graph on Instance Map")
    plt.axis("off")

    # Extract nodes and edges from graph and overlay them
    edges_np = edges.cpu().numpy().T  # Assuming edges is a tensor [2, edges]
    node_positions = []  # List to store positions of nodes

    added_labels = set()
    for idx in range(1, int(instance_map_np.max()) + 1):  # Loop through instance IDs
        # Calculate node position based on the instance map
        node_coords = np.argwhere(instance_map_np == idx)
        if node_coords.size == 0:
            continue
        node_instance = node_coords.mean(axis=0)
        node_positions.append(node_instance)

        # Extract label and corresponding information
        label = int(x_cond_np[instance_map_np == idx].mean())
        label = max(0, min(label, len(OEM_dict) - 1))  # Ensure label is within range
        captions = OEM_dict[label]
        color = np.array(color_palette[label]) / 255.0  # Normalize RGB values to [0, 1]

        if captions not in added_labels:
            plt.scatter(node_instance[1], node_instance[0], color=color, s=20, label=f"{captions}")
            added_labels.add(captions)  # Add label to the set to avoid repetition
        else:
            plt.scatter(node_instance[1], node_instance[0], color=color, s=20)  # No label for duplicate

    # Plot edges based on adjacency using node positions
    for start_idx, end_idx in edges_np:
        if start_idx >= len(node_positions) or end_idx >= len(node_positions):
            continue
        start_position = node_positions[start_idx]
        end_position = node_positions[end_idx]

        # Draw a line (edge) between the start and end positions
        plt.plot([start_position[1], end_position[1]],  # X-coordinates
                 [start_position[0], end_position[0]],  # Y-coordinates
                 color="blue", linewidth=1, alpha=0.7)

    plt.legend(loc="upper left", fontsize='small', bbox_to_anchor=(1.05, 1))
    plt.savefig(save_paths["graph"], bbox_inches='tight', dpi=300)
    plt.close()


def show_graph_with_captions(img_path, node_path, edge_path, instance_map_path, captions_path=None):
    """
    Visualizes the graph, instance map, and image with captions.

    Args:
        img_path (str): Path to the image.
        node_path (str): Path to the file containing graph node data.
        edge_path (str): Path to the file containing graph edge data.
        instance_map_path (str): Path to the `.npy` file containing instance map.
        captions_path (str): Path to the JSON file containing captions data.
    """
    # Read and process segmentation image
    seg = cv2.imread(img_path, cv2.IMREAD_COLOR)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    color_seg = torch.from_numpy(seg)  # Adjust shape

    # Load instance map from .npy
    instance_map = np.load(instance_map_path)

    # Display color segmentation and instance map
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(color_seg)
    plt.title('Color Segmentation')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(instance_map, cmap='jet')
    plt.title('Instance Map')
    plt.colorbar()
    plt.axis('off')

    # Load graph node and edge data
    node = np.load(node_path)
    edge = np.load(edge_path)

    # Create graph data
    graph_data = Data(x=torch.tensor(node, dtype=torch.float), edge_index=torch.tensor(edge, dtype=torch.long))

    # Convert graph data to NetworkX graph
    G = to_networkx(graph_data, to_undirected=True)

    # Create node colors based on the instance map
    node_colors = []
    for i in range(graph_data.x.shape[0]):
        # Find the corresponding instance value for the node
        instance_value = i+1  # Adjust based on your node features
        # Map the instance value to a color from the segmentation image
        mask = (instance_map == instance_value)
        if mask.sum() > 0:
            color = color_seg[mask].float().mean(dim = 0).numpy() / 255.0  # Normalize to [0, 1]
        else:
            color = [0.5, 0.5, 0.5]  # Default color if no matching pixels
        node_colors.append(color)



    # Visualize graph
    plt.subplot(1, 3, 3)
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos,
        node_color=node_colors,
        edge_color='gray',
        node_size=30,
        font_size=6
    )
    plt.title('Graph Visualization', fontsize=12)

    # Add captions below the figure if provided
    if captions_path:
        with open(captions_path, 'r') as f:
            captions = json.load(f)
        plt.figtext(0.5, 0.01, captions.get("caption", ""), wrap=True, horizontalalignment='center', fontsize=8)

    plt.tight_layout()
    plt.show()

def show_noised_image(x, augment_pipe=None, noise_scale=0.09):
    # Sample x_t
    rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
    sigma = (rnd_normal * 1.2 + 1.2).exp()
    weight = (sigma ** 2 + 0.5 ** 2) / (sigma * 0.5) ** 2
    y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

    # Adjusting the noise by the specified noise_scale
    n = noise_scale * torch.randn_like(y) * sigma
    return y + n


def create_patched_image(image, num_patches=25, mask_fraction=0.5 ,mask_indices = None):
    c, h, w = image.shape[1], image.shape[2], image.shape[3]
    patch_size = int(np.sqrt(h * w / num_patches))  # Determine patch size based on num_patches
    patches = []

    # Create patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if i + patch_size <= h and j + patch_size <= w:
                patch = image[:,:, i:i + patch_size, j:j + patch_size]
                patches.append(patch)

    # Randomly mask 50% of the patches
    num_patches = len(patches)
    if mask_indices is  None and mask_fraction > 0.0:
        mask_indices = np.random.choice(num_patches, size=int(num_patches * mask_fraction), replace=False)
    if mask_indices is not None:
        for index in mask_indices:
            patches[index] = torch.full_like(patches[index], fill_value=0.5)  # Set to gray




    # Create a new image from patches
    patched_image = torch.zeros_like(image)
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if idx < len(patches) and i + patch_size <= h and j + patch_size <= w:
                patched_image[:,:, i:i + patch_size, j:j + patch_size] = patches[idx]
                idx += 1

    return patches, patched_image, mask_indices

if __name__ == "__main__":
    import torch

    # img_path = r"D:\dengkai\code\MaskDiT-master\image\patched_noised_image_ori.png"
    #
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
    # image = image / 127.5 - 1.0
    # indices = None
    # patches, patched_single, mask_indices = create_patched_image(image,25,0.0, mask_indices=indices)
    # for i, patched_image in enumerate(patches):
    #
    #     # Convert back to numpy for visualization
    #     patched_image_np = patched_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    #     patched_image_np = (127.5 * (patched_image_np + 1)).astype(np.uint8)  # Scale to [0, 255]
    #
    #     # Save the patched image
    #     output_path = r"D:\dengkai\code\MaskDiT-master\image\noised\noised_patch_{}.png".format(i)
    #
    #     cv2.imwrite(output_path, cv2.cvtColor(patched_image_np, cv2.COLOR_BGR2RGB))
    #
    # img_path = r"D:\dengkai\code\MaskDiT-master\image\kagera_46_0_512.png"
    #
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
    # image = image / 127.5 - 1.0
    #
    # noised_image = show_noised_image(image, None, 5.0)
    #
    # noised_path = r"D:\dengkai\code\MaskDiT-master\image\noised_pure.png"
    #
    # cv2.imwrite(noised_path, cv2.cvtColor((127.5 * (noised_image.squeeze(0).permute(1, 2, 0).detach().numpy() + 1)).astype(np.uint8), cv2.COLOR_BGR2RGB))
    #
    # patches, patched_single, mask_indices = create_patched_image(noised_image)
    #
    #
    # for i,patched_image in enumerate(patches):
    #
    #     # Convert back to numpy for visualization
    #     patched_image_np = patched_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    #     patched_image_np = (127.5 * (patched_image_np + 1)).astype(np.uint8)  # Scale to [0, 255]
    #
    #     # Save the patched image
    #     output_path = r"D:\dengkai\code\MaskDiT-master\image\patched_noised_image_{}.png".format(i)
    #
    #     cv2.imwrite(output_path, cv2.cvtColor(patched_image_np, cv2.COLOR_BGR2RGB))
    #
    #
    # #clip noised image contains 25 patchs,the row and col are 5, and random masked 50%, the masked 50% set as gray
    #
    #
    #
    # #save cliped patchs image contains  masked image and unmasked image
    # patched_single = patched_single.squeeze(0).permute(1, 2, 0).detach().numpy()
    # patched_single = (127.5 * (patched_single + 1)).astype(np.uint8)  # Scale to [0, 255]
    #
    # #Display the patched image
    # plt.imshow(patched_single)
    # plt.axis('off')  # Turn off axis
    # plt.title('Patched Noised Image')
    # plt.show()
    img_path = r"D:\dengkai\code\MaskDiT-master\image\kagera_46_0_512_ref.png"
    node_path = r"D:\dengkai\code\MaskDiT-master\image\graph\kagera_46_0_512_l1.npy"
    edge_path = r"D:\dengkai\code\MaskDiT-master\image\graph\kagera_46_0_512_l1_edge.npy"
    instance_path = r"F:\data\OpenEarthMap\Size_256\val\graph_vit_l_14\instancemap\kagera_46_0_512_inst.npy"
    # captions = r"F:\data\OpenEarthMap\Size_256\train\visual\aachen_6_768_512.json"
    node_save_path = r"D:\dengkai\code\MaskDiT-master\image\graph\kagera_46_0_512_node.png"
    edge_save_path = r"D:\dengkai\code\MaskDiT-master\image\graph\kagera_46_0_512_l1_edge.png"
    instance_save_path = r"D:\dengkai\code\MaskDiT-master\image\graph\kagera_46_0_512_l1_inst.png"

    show_graph_with_captions(img_path, node_path, edge_path,instance_path)