from numba import cuda
import numpy as np
import math
from time import perf_counter
from PIL import Image

@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, width, height, max_iter, image):
    x, y = cuda.grid(2)
    
    if x < width and y < height:
        real = min_x + x * (max_x - min_x) / width
        imag = min_y + y * (max_y - min_y) / height
        c = complex(real, imag)
        z = 0j
        n = 0
        
        while abs(z) <= 2 and n < max_iter:
            z = z * z + c
            n += 1
            
        image[y, x] = n

def create_fractal_gpu(min_x, max_x, min_y, max_y, width, height, max_iter):
    image = np.zeros((height, width), dtype=np.uint8)
    d_image = cuda.to_device(image)
    
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # warm-up (forces JIT compilation and GPU initialization before measurement)
    mandelbrot_kernel[blockspergrid, threadsperblock](
        min_x, max_x, min_y, max_y, width, height, 10, d_image
    )
    cuda.synchronize() # wait for the GPU to finish warming up

    start = perf_counter()

    # real test
    mandelbrot_kernel[blockspergrid, threadsperblock](
        min_x, max_x, min_y, max_y, width, height, max_iter, d_image
    )
    cuda.synchronize()

    end = perf_counter()

    print(f"Exec time (CUDA GPU): {end - start:.4f} sec")
    
    return d_image.copy_to_host()

fractal_image = create_fractal_gpu(-2.0, 1.0, -1.0, 1.0, 2048, 1536, 255)
Image.fromarray(fractal_image).save('mandelbrot_cuda.png')