import os
import random
import matplotlib.pyplot as plt


def plot_30_images_randomly():
    print("plotting 30 images randomly")
    dir_paths = ['TTW(dataset)\\train(576)\\empty', 'TTW(dataset)\\train(576)\\still',
                 'TTW(dataset)\\train(576)\\walking']

    num_rows = 3
    num_cols = 10
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6))
    axes = axes.reshape(-1)
    random.shuffle(axes)

    for i, dir_path in enumerate(dir_paths):
        image_files = os.listdir(dir_path)
        random_images = random.sample(image_files, num_cols)
        for j, random_image in enumerate(random_images):
            image_path = os.path.join(dir_path, random_image)
            img = plt.imread(image_path)
            axes[i * num_cols + j].imshow(img, aspect='auto')
            axes[i * num_cols + j].set_title(f"{os.path.basename(dir_path)}", loc='center')
            axes[i * num_cols + j].axis('off')
            axes[i * num_cols + j].set_frame_on(False)

    fig.suptitle("30 Random Images")
    fig.tight_layout()
    plt.show()
