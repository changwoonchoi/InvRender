import imageio
import numpy as np


SCENES = ["armadillo", "ficus", "hotdog", "lego"]

def gen_mask(img):
    alpha = img[:, :, 3]
    mask = alpha > 128
    return mask

if __name__ == "__main__":
    for scene in SCENES:
        for i in range(100):
            img = imageio.imread(f"../data/tensoir/{scene}/train_{i:03d}/rgba.png")
            mask = gen_mask(img)
            mask_np = mask.astype(np.uint8)
            mask_np *= 255
            imageio.imwrite(f"../data/tensoir/{scene}/train_{i:03d}/mask.png", mask_np)
        for i in range(200):
            img = imageio.imread(f"../data/tensoir/{scene}/test_{i:03d}/rgba.png")
            mask = gen_mask(img)
            mask_np = mask.astype(np.uint8)
            mask_np *= 255
            imageio.imwrite(f"../data/tensoir/{scene}/test_{i:03d}/mask.png", mask_np)


