import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def visualize_3d_array(array_3d, threshold=None, color='blue', alpha=0.3, size=20):
    """
    Visualize a 3D numpy array using matplotlib
    
    Parameters:
    array_3d: 3D numpy array to visualize
    threshold: Only plot points where array value > threshold (if None, plot all non-zero points)
    color: Color of the points
    alpha: Transparency of the points
    size: Size of the points
    """
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of points to plot
    if threshold is not None:
        # Plot points where array value > threshold
        x, y, z = np.where(array_3d > threshold)
        values = array_3d[array_3d > threshold]
    else:
        # Plot all non-zero points
        x, y, z = np.where(array_3d != 0)
        values = array_3d[array_3d != 0]
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', alpha=alpha, s=size)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Array Values')
    
    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set title
    ax.set_title('3D Visualization of Numpy Array')
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    array_3d = np.load('data\\sdf_900.npy')
    array_3d = array_3d.astype(np.float32)
    array_3d = ndimage.rotate(array_3d, angle=-90, axes=(2, 1), reshape=True, order=0)  # Rotate if needed
    
    print(f"Array shape: {array_3d.shape}")
    print(f"Array range: [{array_3d.min():.3f}, {array_3d.max():.3f}]")
    
    visualize_3d_array(array_3d, threshold=0.7, color='red', alpha=0.4, size=30)