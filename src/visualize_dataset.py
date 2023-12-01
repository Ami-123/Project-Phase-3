import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
def show_class_frequency_distribution(output_dir, plots_dir):
    class_labels, num_images = list(), list()
    for folder in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, folder)):
            class_labels += [folder]
            num_images   += [len(os.listdir(os.path.join(output_dir, folder)))]
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, num_images, color='#1f77b4', edgecolor='black', linewidth=1.2)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Class Distribution', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    for i, count in enumerate(num_images):
        plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=12)
    plt.savefig(plots_dir + '/class_distribution.jpeg', dpi=300, bbox_inches='tight')
    #plt.show()

def create_grid_of_sample_images(output_dir, plots_dir):
    class_directories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    rows, cols = 5, 5
    fig_grid, axes = plt.subplots(rows, cols, figsize=(12, 12))
    pixel_intensities = []
    for i in range(rows):
        for j in range(cols):
            class_dir = random.choice(class_directories)
            class_path = os.path.join(output_dir, class_dir)
            image_files = os.listdir(class_path)
            image_file = random.choice(image_files)
            image_path = os.path.join(class_path, image_file)
            img = mpimg.imread(image_path)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_title(class_dir)
            axes[i, j].axis('off')
            pixel_intensities.extend(img.ravel())


    plt.figure(figsize=(10, 6))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Intensity Distribution Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim(0, 256)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the 5x5 grid as an image
    fig_grid.savefig(plots_dir + '/grid_of_images.png')

    # Save the intensity distribution histogram as an image
    plt.savefig(plots_dir + '/intensity_histogram.png')
    #plt.tight_layout()
    #plt.savefig(plots_dir + '/image_grid.jpeg', dpi=300, bbox_inches='tight')
    #plt.show()



def visualize(output_dir, plots_dir):
    show_class_frequency_distribution(output_dir, plots_dir)
    create_grid_of_sample_images(output_dir, plots_dir)