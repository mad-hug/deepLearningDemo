import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_5_images_3rows():
    print("plotting 3 rows with 5 images each")

    # Set the directory paths
    dir_paths = ['TTW(dataset)\\train(576)\\empty', 'TTW(dataset)\\train(576)\\still',
                 'TTW(dataset)\\train(576)\\walking']

    # Set the number of images to display per row
    num_images = 5

    # Set the row titles
    row_titles = ['Empty', 'Still', 'Walking']

    # Create the figure and axis objects
    fig, axes = plt.subplots(nrows=len(dir_paths), ncols=num_images + 1, figsize=(20, 10))

    # Loop through each directory
    for i, dir_path in enumerate(dir_paths):
        # Set the title for the current row
        title = row_titles[i]

        # Add the title to the first column of the current row
        axes[i, 0].set_title(title, loc='center', fontweight='bold', fontstyle='italic', pad=20, va='center')

        # Get a list of random image file names from the current directory
        image_names = os.listdir(dir_path)
        image_names = random.sample(image_names, num_images)

        # Loop through each image and display it on the corresponding axis
        for j, image_name in enumerate(image_names):
            # Get the full path of the image
            image_path = os.path.join(dir_path, image_name)

            # Load the image and display it on the corresponding axis
            image = plt.imread(image_path)
            axes[i, j + 1].imshow(image, aspect='auto')
            axes[i, j + 1].axis('off')

        # Remove the borders and ticks from the current row
        for ax in axes[i]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, left=False)

    # Remove the borders and ticks from the first column
    for ax in axes[:, 0]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, left=False)
        ax.axis('off')

    # Set the spacing between the images in the same row
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    # Show the plot
    plt.show()