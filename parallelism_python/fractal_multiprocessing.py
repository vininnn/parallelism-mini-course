import numpy as np
from PIL import Image
from time import perf_counter
from multiprocessing import Pool, cpu_count

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

def create_fractal_multiprocessing(min_x, max_x, min_y, max_y, width, height, max_iter):
    img = np.zeros((height, width), dtype=np.uint8)
    tasks = [(y, min_x, max_x, min_y, max_y, width, height, max_iter) for y in range(height)]

    #  distributes image rows among available cores
    with Pool(processes=cpu_count()) as pool:
        for y, row in pool.imap_unordered(compute_row, tasks):
            img[y] = row

    return img

if __name__ == "__main__":
    start = perf_counter()

    fractal_image = create_fractal_multiprocessing(-2.0, 1.0, -1.0, 1.0, 2048, 1536, 255)

    end = perf_counter()

    time = end - start

    print(f"Exec time: {time:.4f} sec")

    Image.fromarray(fractal_image).save('mandelbrot_multiprocessing.png')