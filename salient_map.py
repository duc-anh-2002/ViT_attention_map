import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define a function to preprocess the image
def preprocess_image(img_path, img_size=224):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img, img_tensor

# Define a function to compute the saliency map
def compute_saliency_map(model, img_tensor):
    img_tensor.requires_grad_()  # Allow gradients for input image
    output = model(img_tensor)  # Forward pass
    
    # Get the index of the class with the highest score (predicted class)
    predicted_class = output.argmax().item()
    
    # Backward pass to compute gradients with respect to the predicted class
    model.zero_grad()
    output[0, predicted_class].backward()
    
    # Get the saliency map (absolute value of gradients)
    saliency_map = img_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency_map = np.max(saliency_map, axis=0)  # Take the maximum over the RGB channels
    return saliency_map

# Define a function to plot and save the original image and saliency map
def plot_and_save_saliency(original_img, saliency_map, save_path="saliency_map.png"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot saliency map
    axs[1].imshow(saliency_map, cmap='hot')
    axs[1].axis('off')
    axs[1].set_title('Saliency Map')

    # Save the saliency map
    plt.savefig(save_path)
    plt.show()

# Example Usage
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    img_path = '/home/ubuntu/workspace/ViT_Attention_Map_Visualization/Pic3.png'  # Replace with your image path

    # Preprocess the image and get its tensor representation
    original_img, img_tensor = preprocess_image(img_path)

    # print(img_tensor) # torch.Size([1, 3, 224, 224])
    # Compute the saliency map
    saliency_map = compute_saliency_map(model, img_tensor) # (224, 224)

    # Plot and save the saliency map
    plot_and_save_saliency(original_img, saliency_map, save_path="saliency_map.png")
    
    # Save the saliency matrix as a .npy file (numpy array format)
    np.save('saliency_matrix.npy', saliency_map)

    print(np.array(saliency_map))