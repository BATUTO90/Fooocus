import torch
import numpy as np

from PIL import Image, ImageFilter
from modules.util import resample_image, set_image_shape_ceil, get_image_shape_ceil
from modules.upscaler import perform_upscale
import cv2


inpaint_head_model = None


class InpaintHead(torch.nn.Module):
    """A PyTorch module that represents the inpainting head."""
    def __init__(self, *args, **kwargs):
        """Initializes the InpaintHead."""
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device='cpu'))

    def __call__(self, x):
        """Performs the forward pass of the inpainting head.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)


current_task = None


def box_blur(x, k):
    """Applies a box blur to an image.
    Args:
        x (np.ndarray): The input image.
        k (int): The kernel size.
    Returns:
        np.ndarray: The blurred image.
    """
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.BoxBlur(k))
    return np.array(x)


def max_filter_opencv(x, ksize=3):
    """Applies a maximum filter to an image using OpenCV.
    Args:
        x (np.ndarray): The input image.
        ksize (int, optional): The kernel size. Defaults to 3.
    Returns:
        np.ndarray: The filtered image.
    """
    # Use OpenCV maximum filter
    # Make sure the input type is int16
    return cv2.dilate(x, np.ones((ksize, ksize), dtype=np.int16))


def morphological_open(x):
    """Performs a morphological opening on an image.
    Args:
        x (np.ndarray): The input image.
    Returns:
        np.ndarray: The opened image.
    """
    # Convert array to int16 type via threshold operation
    x_int16 = np.zeros_like(x, dtype=np.int16)
    x_int16[x > 127] = 256

    for i in range(32):
        # Use int16 type to avoid overflow
        maxed = max_filter_opencv(x_int16, ksize=3) - 8
        x_int16 = np.maximum(maxed, x_int16)

    # Clip negative values to 0 and convert back to uint8 type
    x_uint8 = np.clip(x_int16, 0, 255).astype(np.uint8)
    return x_uint8


def up255(x, t=0):
    """Converts an image to a binary image with values of 0 or 255.
    Args:
        x (np.ndarray): The input image.
        t (int, optional): The threshold value. Defaults to 0.
    Returns:
        np.ndarray: The binary image.
    """
    y = np.zeros_like(x).astype(np.uint8)
    y[x > t] = 255
    return y


def imsave(x, path):
    """Saves an image to a file.
    Args:
        x (np.ndarray): The image to save.
        path (str): The path to the file.
    """
    x = Image.fromarray(x)
    x.save(path)


def regulate_abcd(x, a, b, c, d):
    """Regulates the coordinates of a bounding box to be within the image boundaries.
    Args:
        x (np.ndarray): The input image.
        a (int): The top coordinate.
        b (int): The bottom coordinate.
        c (int): The left coordinate.
        d (int): The right coordinate.
    Returns:
        tuple[int, int, int, int]: The regulated coordinates.
    """
    H, W = x.shape[:2]
    if a < 0:
        a = 0
    if a > H:
        a = H
    if b < 0:
        b = 0
    if b > H:
        b = H
    if c < 0:
        c = 0
    if c > W:
        c = W
    if d < 0:
        d = 0
    if d > W:
        d = W
    return int(a), int(b), int(c), int(d)


def compute_initial_abcd(x):
    """Computes the initial bounding box of a mask.
    Args:
        x (np.ndarray): The input mask.
    Returns:
        tuple[int, int, int, int]: The initial bounding box.
    """
    indices = np.where(x)
    a = np.min(indices[0])
    b = np.max(indices[0])
    c = np.min(indices[1])
    d = np.max(indices[1])
    abp = (b + a) // 2
    abm = (b - a) // 2
    cdp = (d + c) // 2
    cdm = (d - c) // 2
    l = int(max(abm, cdm) * 1.15)
    a = abp - l
    b = abp + l + 1
    c = cdp - l
    d = cdp + l + 1
    a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def solve_abcd(x, a, b, c, d, k):
    """Solves for the bounding box of a mask.
    Args:
        x (np.ndarray): The input mask.
        a (int): The top coordinate.
        b (int): The bottom coordinate.
        c (int): The left coordinate.
        d (int): The right coordinate.
        k (float): The scaling factor.
    Returns:
        tuple[int, int, int, int]: The solved bounding box.
    """
    k = float(k)
    assert 0.0 <= k <= 1.0

    H, W = x.shape[:2]
    if k == 1.0:
        return 0, H, 0, W
    while True:
        if b - a >= H * k and d - c >= W * k:
            break

        add_h = (b - a) < (d - c)
        add_w = not add_h

        if b - a == H:
            add_w = True

        if d - c == W:
            add_h = True

        if add_h:
            a -= 1
            b += 1

        if add_w:
            c -= 1
            d += 1

        a, b, c, d = regulate_abcd(x, a, b, c, d)
    return a, b, c, d


def fooocus_fill(image, mask):
    """Fills an image using the Fooocus inpainting algorithm.
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The inpainting mask.
    Returns:
        np.ndarray: The filled image.
    """
    current_image = image.copy()
    raw_image = image.copy()
    area = np.where(mask < 127)
    store = raw_image[area]

    for k, repeats in [(512, 2), (256, 2), (128, 4), (64, 4), (33, 8), (15, 8), (5, 16), (3, 16)]:
        for _ in range(repeats):
            current_image = box_blur(current_image, k)
            current_image[area] = store

    return current_image


class InpaintWorker:
    """A class that performs inpainting on an image."""
    def __init__(self, image, mask, use_fill=True, k=0.618):
        """Initializes the InpaintWorker.
        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): The inpainting mask.
            use_fill (bool, optional): Whether to use filling. Defaults to True.
            k (float, optional): The scaling factor for the bounding box. Defaults to 0.618.
        """
        a, b, c, d = compute_initial_abcd(mask > 0)
        a, b, c, d = solve_abcd(mask, a, b, c, d, k=k)

        # interested area
        self.interested_area = (a, b, c, d)
        self.interested_mask = mask[a:b, c:d]
        self.interested_image = image[a:b, c:d]

        # super resolution
        if get_image_shape_ceil(self.interested_image) < 1024:
            self.interested_image = perform_upscale(self.interested_image)

        # resize to make images ready for diffusion
        self.interested_image = set_image_shape_ceil(self.interested_image, 1024)
        self.interested_fill = self.interested_image.copy()
        H, W, C = self.interested_image.shape

        # process mask
        self.interested_mask = up255(resample_image(self.interested_mask, W, H), t=127)

        # compute filling
        if use_fill:
            self.interested_fill = fooocus_fill(self.interested_image, self.interested_mask)

        # soft pixels
        self.mask = morphological_open(mask)
        self.image = image

        # ending
        self.latent = None
        self.latent_after_swap = None
        self.swapped = False
        self.latent_mask = None
        self.inpaint_head_feature = None
        return

    def load_latent(self, latent_fill, latent_mask, latent_swap=None):
        """Loads the latent variables.
        Args:
            latent_fill: The latent variable for the fill.
            latent_mask: The latent variable for the mask.
            latent_swap (optional): The latent variable for the swap. Defaults to None.
        """
        self.latent = latent_fill
        self.latent_mask = latent_mask
        self.latent_after_swap = latent_swap
        return

    def patch(self, inpaint_head_model_path, inpaint_latent, inpaint_latent_mask, model):
        """Patches the model with the inpainting head.
        Args:
            inpaint_head_model_path (str): The path to the inpainting head model.
            inpaint_latent: The inpainting latent variable.
            inpaint_latent_mask: The inpainting latent mask.
            model: The model to patch.
        Returns:
            The patched model.
        """
        global inpaint_head_model

        if inpaint_head_model is None:
            inpaint_head_model = InpaintHead()
            sd = torch.load(inpaint_head_model_path, map_location='cpu', weights_only=True)
            inpaint_head_model.load_state_dict(sd)

        feed = torch.cat([
            inpaint_latent_mask,
            model.model.process_latent_in(inpaint_latent)
        ], dim=1)

        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        inpaint_head_feature = inpaint_head_model(feed)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                h = h + inpaint_head_feature.to(h)
            return h

        m = model.clone()
        m.set_model_input_block_patch(input_block_patch)
        return m

    def swap(self):
        """Swaps the latent variables."""
        if self.swapped:
            return

        if self.latent is None:
            return

        if self.latent_after_swap is None:
            return

        self.latent, self.latent_after_swap = self.latent_after_swap, self.latent
        self.swapped = True
        return

    def unswap(self):
        """Unswaps the latent variables."""
        if not self.swapped:
            return

        if self.latent is None:
            return

        if self.latent_after_swap is None:
            return

        self.latent, self.latent_after_swap = self.latent_after_swap, self.latent
        self.swapped = False
        return

    def color_correction(self, img):
        """Performs color correction on an image.
        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: The color-corrected image.
        """
        fg = img.astype(np.float32)
        bg = self.image.copy().astype(np.float32)
        w = self.mask[:, :, None].astype(np.float32) / 255.0
        y = fg * w + bg * (1 - w)
        return y.clip(0, 255).astype(np.uint8)

    def post_process(self, img):
        """Post-processes an image.
        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: The post-processed image.
        """
        a, b, c, d = self.interested_area
        content = resample_image(img, d - c, b - a)
        result = self.image.copy()
        result[a:b, c:d] = content
        result = self.color_correction(result)
        return result

    def visualize_mask_processing(self):
        """Visualizes the mask processing.
        Returns:
            list[np.ndarray]: A list of images that represent the mask processing steps.
        """
        return [self.interested_fill, self.interested_mask, self.interested_image]

