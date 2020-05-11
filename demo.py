import click

@click.command()
@click.argument('model_path', type=click.Path(readable=True))
@click.option('--image_path', type=click.Path(readable=True))
@click.option('--device', type=str)
def demo(model_path, image_path, device):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.ndimage import sobel
    from PIL import Image
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    torch_device = torch.device(device)

    # Load model
    dextr_model = torch.load(model_path, map_location=torch_device)
    dextr_model.eval()

    # Load image
    if image_path is None:
        image_path = os.path.join('images', 'giraffes_1.jpg')
    image = Image.open(image_path)
    img_arr = np.array(image) / 255.0
    plt.ion()
    plt.axis('off')
    plt.imshow(img_arr)
    plt.title('Click four extreme points of the object\nHit enter when done')

    def overlay(image, colour, mask, alpha):
        alpha_img = mask * alpha
        return image * (1 - alpha_img[:, :, None]) + colour[None, None, :] * alpha_img[:, :, None]

    while True:
        # Get points from user
        extreme_points = np.array(plt.ginput(4, timeout=0))
        if len(extreme_points) < 4:
            # Less than four points; quit
            break

        # Predict masks (points come from matplotlib in [x,y] order; this must be flipped)
        masks = dextr_model.predict([image], extreme_points[None, :, ::-1], torch_device)

        mask_bin = masks[0] >= 0.5
        edges = sobel(mask_bin.astype(float)) != 0

        img_arr = overlay(img_arr, np.array([0.0, 1.0, 0.0]), mask_bin, 0.3)
        img_arr = overlay(img_arr, np.array([1.0, 1.0, 0.0]), edges, 0.3)

        plt.imshow(img_arr)
        plt.plot(extreme_points[:, 0], extreme_points[:, 1], 'gx')


if __name__ == '__main__':
    demo()

