include(joinpath("..","src","DRL.jl"))
using DRL
using MXNet

# TODO all of this stolen from test_CG.jl -- modify to some simply canonical TRO problem
# TODO use a schematic form of the Value update in the GAE paper
# Pretend simple quadratic form for value function
n = 5

A = rand(n,n)
#b = rand(n)
x = rand(n)

# estimate of value
v_estimate = (x'*A*x)[1] #(x'*A*x + b'*x)[1]
td = 2
v_target = v_estimate + td

# gradient of estimator
grad_est = vec( x*x' )

# gradient of the objective
grad_obj = td * grad_est

# first order approx of hessian as 'A' matrix
H = grad_est*grad_est'
_A = mx.copy!(mx.zeros(n*n,n*n), H)

# gradient of objective as 'b' matrix
_b = mx.copy!(mx.zeros(n*n,1), reshape(grad_obj, n*n, 1))

# define objective function (in mxnet context)
th0 = mx.copy!(mx.zeros(n*n,1), reshape(A, n*n,1) )
_grad_obj = mx.copy!(mx.zeros(n*n,1), reshape(grad_obj, n*n,1))
obj( th ) = copy!(zeros(Float32,1,1),mx.dot(_grad_obj',(th - th0)))[1]

# define constraint function
constr( th, th_old ) = mx.copy!(zeros(Float32,1,1), mx.dot((th - th_old)',mx.dot(_A,(th - th_old))))[1]

# solve problem
_thp = rl.trust_region(obj, constr, th0, rl.DenseMatrix(_A), _b)
thp = mx.copy!(zeros(Float32,n*n,1), _thp)
# 
v_estimate_ = (x'*reshape(thp,n,n)*x)[1]
td_ = abs(v_target - v_estimate_)
println("Old Error: $(td), New Error: $(td_)")
assert( td_ < td )

