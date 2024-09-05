import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import numpy as np
import sys

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans
        
        # Linear layer to embed each patch
        self.projection = nn.Linear(self.patch_dim, embed_dim)
    
    def forward(self, x):
        # Extract patches from the image
        batch_size, channels, height, width = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, self.patch_dim)
        
        # Linear projection of patches to embedding space
        embeddings = self.projection(patches)
        return embeddings

class AttentionMap(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5  # Scaling factor
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Linear projections for Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores (Q * K^T)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_map = F.softmax(attention_scores, dim=-1)  # Softmax along the patch dimension
        
        # Compute the final output by applying attention map to values
        output = torch.matmul(attention_map, V)
        return attention_map, output

def plot_images(original_img, attention_map, num_patches_side):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original image
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Plot the attention map
    attention_map_agg = attention_map.mean(0).detach().numpy()  # Average over heads
    attention_map_2d = attention_map_agg.reshape(num_patches_side, num_patches_side)  # Reshape into 2D grid
    
    axs[1].imshow(attention_map_2d, cmap="inferno")
    axs[1].set_title("Attention Map")
    axs[1].axis("off")

    #plt.show()
    plt.savefig('./output.png')


def process_image(img_path, img_size=32):
    # Load image
    img = Image.open(img_path).convert('RGB')

    # Transform: resize and convert to tensor
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img, img_tensor

# Example Usage
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    img_path = '/home/ubuntu/workspace/ViT_Attention_Map_Visualization/Pic3.png'  # Replace with your image path

    # Load and process image
    original_img, img_tensor = process_image(img_path, img_size=32)

    # Define patch embedding and attention modules
    patch_embed = PatchEmbedding(img_size=32, patch_size=4, embed_dim=64)
    attention_layer = AttentionMap(embed_dim=64)

    # Get patch embeddings
    patch_embeddings = patch_embed(img_tensor)  # Shape: (batch_size, num_patches, embed_dim)
    
    # Compute attention map
    attention_map, _ = attention_layer(patch_embeddings)

    # Number of patches per side
    num_patches_side = int(32 // 4)  # 32x32 image with 4x4 patches -> 8x8 patches

    # Plot original image and attention map
    plot_images(original_img, attention_map[0], num_patches_side=num_patches_side)
    #import ipdb; ipdb.set_trace()
    print(attention_map[0].detach().numpy())
    
