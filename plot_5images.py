import os
import random
import matplotlib.pyplot as plt


def plot_5_images_3rows():
    print("plotting 3 rows with 5 images each")

    dir_paths = ['TTW(dataset)\\train(576)\\empty', 'TTW(dataset)\\train(576)\\still',
                 'TTW(dataset)\\train(576)\\walking']

    num_images = 5

    row_titles = ['Empty', 'Still', 'Walking']

    fig, axes = plt.subplots(nrows=len(dir_paths), ncols=num_images + 1)

    for i, dir_path in enumerate(dir_paths):
        title = row_titles[i]
        axes[i, 0].set_title(title)

        image_names = os.listdir(dir_path)
        image_names = random.sample(image_names, num_images)

        for j, image_name in enumerate(image_names):
            image_path = os.path.join(dir_path, image_name)

            image = plt.imread(image_path)
            axes[i, j + 1].imshow(image, aspect='auto')
            axes[i, j + 1].axis('off')

    # Remove the borders
    for ax in axes[:, 0]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, left=False)
        ax.axis('off')

    # Show the plot
    plt.show()