# Libs >>>
import torch
import random
from torchvision.transforms import v2 as v2_transforms


# Func >>>
def get_scaling_random_crop(magnitude, img_size):
    # Calculate crop size based on magnitude
    min_crop_scale = 0.65
    max_crop_scale = 0.95
    crop_scale = max_crop_scale - (magnitude / 30) * (max_crop_scale - min_crop_scale)

    # Calculate padding size based on magnitude
    max_padding_scale = 0.25
    padding_scale = (magnitude / 30) * max_padding_scale

    # Calculate aspect ratio range based on magnitude
    min_aspect_ratio = 0.75
    max_aspect_ratio = 1.0 / min_aspect_ratio
    aspect_ratio_range = (
        max(min_aspect_ratio, 1.0 - (magnitude / 30) * (1.0 - min_aspect_ratio)),
        min(max_aspect_ratio, 1.0 + (magnitude / 30) * (max_aspect_ratio - 1.0)),
    )

    return v2_transforms.Compose(
        [
            v2_transforms.RandomCrop(
                size=[int(s * crop_scale) for s in img_size],
                padding=[int(s * padding_scale) for s in img_size],
                pad_if_needed=True,
                padding_mode="symmetric",
                fill=None,
            ),
            v2_transforms.RandomResizedCrop(
                size=img_size,
                scale=(crop_scale**2, 1.0),
                ratio=aspect_ratio_range,
                antialias=True,
            ),
        ]
    )


def rgb_augmentation_transform(magnitude=10, img_size=(224, 224)):
    # Ensure magnitude is within the range
    magnitude = max(0.5, min(magnitude, 30))

    # Calculate probabilities for static transforms
    prob = min(magnitude / 12.0, 0.5)

    Augment_transformes = [
        # Geometric transforms ----------------------------------------------
        v2_transforms.RandomVerticalFlip(p=prob),
        v2_transforms.RandomHorizontalFlip(p=prob),
        v2_transforms.RandomRotation(
            degrees=magnitude * 4,
            expand=False,
            fill=(
                random.randint(25, 230),
                random.randint(25, 230),
                random.randint(25, 230),
            ),
        ),
        # v2_transforms.ElasticTransform(alpha=magnitude / 2, sigma=12 - magnitude / 4.4), # Makes the process slower
        v2_transforms.ScaleJitter(
            target_size=img_size, scale_range=(1 - (magnitude / 60), 1)
        ),
        v2_transforms.RandomApply(
            torch.nn.ModuleList([get_scaling_random_crop(magnitude, img_size)]), p=prob
        ),
        # Photometric transforms ----------------------------------------------
        v2_transforms.RandomErasing(
            scale=(0.005, 0.005 + magnitude / 512), value="random", p=prob
        ),
        v2_transforms.RandomInvert(p=prob),
        v2_transforms.RandomApply(
            torch.nn.ModuleList([v2_transforms.RandomChannelPermutation()]), p=prob
        ),
        # v2_transforms.RandomGrayscale(p=prob), # Not recommended for some datasets
        v2_transforms.RandomAdjustSharpness(
            1 + max(random.random() - (1 - (magnitude / 30)), 0), p=prob
        ),
        v2_transforms.GaussianNoise(sigma=magnitude / 200.0),
        v2_transforms.GaussianBlur(
            kernel_size=(
                [3, 5, 7, 9, 11][int(magnitude / 7.5)],
                [3, 5, 7, 9, 11][int(magnitude / 7.5)],
            ),
            sigma=(0.05, 0.05 + magnitude / 17.0),
        ),
    ]
    # Main transform list
    transform_list = [
        # Prep data for augmentation
        v2_transforms.ToDtype(torch.float32),
        # Augmentations
        v2_transforms.Compose(Augment_transformes),
        # Resize to original size
        v2_transforms.Resize(img_size, antialias=True),
    ]
    # End
    return v2_transforms.Compose(transform_list)
