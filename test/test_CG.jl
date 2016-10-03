include(joinpath("..","src","DRL.jl"))
using DRL
using MXNet

n = 5
# Generate a symmetric positive definite matrix
A = rand(n,n)
A = 0.5 * (A + A')
A += n*eye(n)
# via stack overflow: http://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab

b = rand(n)

# analytical solution
x_exact = A\b

# move everything to the MXNet context
_A = mx.copy!(mx.zeros(n,n), A)
_b = mx.copy!(mx.zeros(n,1), reshape(b, n, 1)) # MXNet is hateful like that

x_approx = rl.conjugate_gradient(rl.DenseMatrix(_A), _b)

# move back to julia context
x_approx = vec( mx.copy!(zeros(Float32, n, 1), x_approx) )
println(norm(x_exact - x_approx))
assert( norm(x_exact - x_approx) < 0.0001)

