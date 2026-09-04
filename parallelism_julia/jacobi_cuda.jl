using CUDA
function jacobi_kernel_gpu!(new_array, old_array, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 1 < i < n
        @inbounds new_array[i] = (old_array[i-1] + old_array[i+1]) * 0.5f0
    end
    return
end

function run_jacobi_gpu(size, iterations)
    a = zeros(Float32, size)
    a[1] = 100.0f0
    a[end] = 100.0f0

    a_d = CuArray(a)
    next_a_d = copy(a_d)

    threads = 256
    blocks = cld(size, threads)

    for _ in 1:iterations
        @cuda threads=threads blocks=blocks jacobi_kernel_gpu!(next_a_d, a_d, size)
        a_d, next_a_d = next_a_d, a_d
    end

    CUDA.synchronize()

    return Array(a_d)
end

const SIZE = 10_000_000
const ITERATIONS = 1000

run_jacobi_gpu(128, 1)

CUDA.@time final_state = run_jacobi_gpu(SIZE, ITERATIONS)

println("Resultado no centro (GPU): ", final_state[div(SIZE, 2)])