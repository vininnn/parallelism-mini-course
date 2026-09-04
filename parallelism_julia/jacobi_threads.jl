using Base.Threads
function jacobi_1d_kernel_threads!(new_array, old_array)
    n = length(old_array)
    @threads for i in 2:n-1
        @inbounds new_array[i] = (old_array[i-1] + old_array[i+1]) / 2.0
    end
end

function run_jacobi(size, iterations)
    a = zeros(Float64, size)
    next_a = zeros(Float64, size)

	a[1] = 100.0
	a[size] = 100.0
	next_a[1] = 100.0
	next_a[size] = 100.0

	for _ in 1:iterations
		jacobi_1d_kernel_threads!(next_a, a)
		a, next_a = next_a, a
	end
	return a
end

const SIZE = 10_000_000
const ITERATIONS = 1000
println("Available Threads:", Threads.nthreads())

run_jacobi(10, 1)

@time final_state = run_jacobi(SIZE, ITERATIONS)
println("Resultado no centro do vetor: ", final_state[div(SIZE, 2)])
println("Resultado em um quarto do vetor: ", final_state[div(SIZE, 4)])