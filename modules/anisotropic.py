import torch


Tensor = torch.Tensor
Device = torch.DeviceObjType
Dtype = torch.Type
pad = torch.nn.functional.pad


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """Computes the zero padding for a given kernel size.
    Args:
        kernel_size (tuple[int, int] or int): The kernel size.
    Returns:
        tuple[int, int]: The zero padding.
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """Unpacks a 2D kernel size into its y and x components.
    Args:
        kernel_size (tuple[int, int] or int): The kernel size.
    Returns:
        tuple[int, int]: The y and x components of the kernel size.
    """
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, '2D Kernel size should have a length of 2.'
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def gaussian(
    window_size: int, sigma: Tensor | float, *, device: Device | None = None, dtype: Dtype | None = None
) -> Tensor:
    """Generates a 1D Gaussian kernel.
    Args:
        window_size (int): The size of the window.
        sigma (Tensor or float): The standard deviation of the Gaussian.
        device (Device, optional): The device to create the tensor on. Defaults to None.
        dtype (Dtype, optional): The data type of the tensor. Defaults to None.
    Returns:
        Tensor: The 1D Gaussian kernel.
    """

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:
    """Generates a 1D Gaussian kernel.
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float or Tensor): The standard deviation of the Gaussian.
        force_even (bool, optional): Whether to force the kernel to be even. Defaults to False.
        device (Device, optional): The device to create the tensor on. Defaults to None.
        dtype (Dtype, optional): The data type of the tensor. Defaults to None.
    Returns:
        Tensor: The 1D Gaussian kernel.
    """

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:
    """Generates a 2D Gaussian kernel.
    Args:
        kernel_size (tuple[int, int] or int): The size of the kernel.
        sigma (tuple[float, float] or Tensor): The standard deviation of the Gaussian.
        force_even (bool, optional): Whether to force the kernel to be even. Defaults to False.
        device (Device, optional): The device to create the tensor on. Defaults to None.
        dtype (Dtype, optional): The data type of the tensor. Defaults to None.
    Returns:
        Tensor: The 2D Gaussian kernel.
    """

    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:

    if isinstance(sigma_color, Tensor):
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: tuple[float, float] | Tensor = 3.0,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """Applies a bilateral blur to an image.
    Args:
        input (Tensor): The input image.
        kernel_size (tuple[int, int] or int, optional): The size of the kernel. Defaults to (13, 13).
        sigma_color (float or Tensor, optional): The standard deviation of the color space. Defaults to 3.0.
        sigma_space (tuple[float, float] or Tensor, optional): The standard deviation of the coordinate space. Defaults to 3.0.
        border_type (str, optional): The padding mode. Defaults to 'reflect'.
        color_distance_type (str, optional): The color distance type. Defaults to 'l1'.
    Returns:
        Tensor: The blurred image.
    """
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def adaptive_anisotropic_filter(x, g=None):
    """Applies an adaptive anisotropic filter to an image.
    Args:
        x (Tensor): The input image.
        g (Tensor, optional): The guidance image. Defaults to None.
    Returns:
        Tensor: The filtered image.
    """
    if g is None:
        g = x
    s, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)
    s = s + 1e-5
    guidance = (g - m) / s
    y = _bilateral_blur(x, guidance,
                        kernel_size=(13, 13),
                        sigma_color=3.0,
                        sigma_space=3.0,
                        border_type='reflect',
                        color_distance_type='l1')
    return y


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    """Applies a joint bilateral blur to an image.
    Args:
        input (Tensor): The input image.
        guidance (Tensor): The guidance image.
        kernel_size (tuple[int, int] or int): The size of the kernel.
        sigma_color (float or Tensor): The standard deviation of the color space.
        sigma_space (tuple[float, float] or Tensor): The standard deviation of the coordinate space.
        border_type (str, optional): The padding mode. Defaults to 'reflect'.
        color_distance_type (str, optional): The color distance type. Defaults to 'l1'.
    Returns:
        Tensor: The blurred image.
    """
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


class _BilateralBlur(torch.nn.Module):
    """Base class for bilateral blur modules."""
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: str = 'reflect',
        color_distance_type: str = "l1",
    ) -> None:
        """Initializes the _BilateralBlur module.
        Args:
            kernel_size (tuple[int, int] or int): The size of the kernel.
            sigma_color (float or Tensor): The standard deviation of the color space.
            sigma_space (tuple[float, float] or Tensor): The standard deviation of the coordinate space.
            border_type (str, optional): The padding mode. Defaults to 'reflect'.
            color_distance_type (str, optional): The color distance type. Defaults to 'l1'.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    """Applies a bilateral blur to an image."""
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the BilateralBlur module.
        Args:
            input (Tensor): The input image.
        Returns:
            Tensor: The blurred image.
        """
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )


class JointBilateralBlur(_BilateralBlur):
    """Applies a joint bilateral blur to an image."""
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        """Forward pass of the JointBilateralBlur module.
        Args:
            input (Tensor): The input image.
            guidance (Tensor): The guidance image.
        Returns:
            Tensor: The blurred image.
        """
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )
