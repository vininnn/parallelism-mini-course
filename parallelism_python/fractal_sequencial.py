import numpy as np
from PIL import Image
from time import perf_counter

def mandelbrot_kernel(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def create_fractal(min_x, max_x, min_y, max_y, width, height, max_iter):
    img = np.zeros((height, width), dtype=np.uint8)
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        for y in range(height):
            real = min_x + x * pixel_size_x
            imag = min_y + y * pixel_size_y
            color = mandelbrot_kernel(complex(real, imag), max_iter)
            img[y, x] = color

    return img

start = perf_counter()

fractal_image = create_fractal(-2.0, 1.0, -1.0, 1.0, 2048, 1536, 255)

end = perf_counter()

time = end - start

print(f"Exec time: {time:.4f} sec")

Image.fromarray(fractal_image).save('mandelbrot_sequential.png')
 