import numpy as np
from numba import njit, prange
from PIL import Image
from time import perf_counter

@njit
def mandelbrot_kernel(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def compute_row(args):
    y, min_x, max_x, min_y, max_y, width, height, max_iter = args
    row = np.zeros(width, dtype=np.uint8)
    pixel_size_x = (max_x - min_x) / width
    imag = min_y + y * (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        row[x] = mandelbrot_kernel(complex(real, imag), max_iter)
    return y, row

@njit(parallel=True)
def create_fractal_numba_cpu(min_x, max_x, min_y, max_y, width, height, max_iter):
    img = np.zeros((height, width), dtype=np.uint8)
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    # prange, instead of range, distributes loops across CPU threads.
    for y in prange(height):
        imag = min_y + y * pixel_size_y
        for x in range(width):
            real = min_x + x * pixel_size_x
            color = mandelbrot_kernel(complex(real, imag), max_iter)
            img[y, x] = color

    return img

# warm-up
# the first Numba call includes the JIT compilation time
# we make a quick call beforehand to compile it
_ = create_fractal_numba_cpu(-2.0, 1.0, -1.0, 1.0, 100, 100, 10)

start = perf_counter()

fractal_image = create_fractal_numba_cpu(-2.0, 1.0, -1.0, 1.0, 2048, 1536, 255)

end = perf_counter()

print(f"Exec time (Numba CPU): {end - start:.4f} sec")

Image.fromarray(fractal_image).save('mandelbrot_numba.png')

# from numba import njit, prange
# import numpy as np
# from PIL import Image

# @njit(parallel=True)
# def create_fractal_numba_cpu(min_x, max_x, min_y, max_y, width, height, max_iter):
#     img = np.zeros((height, width), dtype=np.uint8)
#     pixel_size_x = (max_x - min_x) / width
#     pixel_size_y = (max_y - min_y) / height

#     # prange, instead of range, distributes loops across CPU threads.
#     for y in prange(height):
#         imag = min_y + y * pixel_size_y
#         for x in range(width):
#             real = min_x + x * pixel_size_x
#             c = complex(real, imag)
#             z = 0j
#             n = 0
#             while abs(z) <= 2 and n < max_iter:
#                 z = z * z + c
#                 n += 1
#             img[y, x] = n

#     return img

# fractal_image = create_fractal_numba_cpu(-2.0, 1.0, -1.0, 1.0, 2048, 1536, 255)
# Image.fromarray(fractal_image).save('mandelbrot_numba_cpu.png')