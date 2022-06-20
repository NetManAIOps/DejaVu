from typing import Tuple, Union


def conv_2d_output_shape(
        h_w: Tuple, kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: int = 1, pad: int = 0, dilation: int = 1
) -> Tuple[int, int]:
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[-2] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[-1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def conv_1d_output_shape(
        length: int, kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: int = 1, pad: int = 0, dilation: int = 1
) -> int:
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    length = floor(((length + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    return length