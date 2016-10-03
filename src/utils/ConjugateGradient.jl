# implementation of Conjugate gradient algorithm for optimization (also Trust Region)

# generic representation for something that you can get a matrix-vector product out of
abstract FactoredMatrix

#mult!(::FactoredMatrix, x::Union{Vector{Real}, Matrix{Real}} ) = error("Undefined")

type DenseMatrix <: FactoredMatrix
    A::Union{Matrix{Real},mx.NDArray}
end
mult!(dm::DenseMatrix, x::Union{Vector{Real},mx.NDArray,Matrix{Real}}) = isa(x,mx.NDArray) ? mx.dot(dm.A,x) : A*x

# NOTE: size(kl_hessian)=(|A|,|A|, N), size(jacobian)=(|th|,|A|,N) A = action space, th = param space, N = nb samples
type FisherInformation <: FactoredMatrix
    kl_hessian::Union{Matrix{Real},mx.NDArray}
    jacobian::Union{Matrix{Real},mx.NDArray}
    subsample::Float64 # to get an approx of aprox expectation
    rng::Union{AbstractRNG,Void}
    result::Union{mx.NDArray,Vector{Real},Matrix{Real},Void}
end

function mult!(fim::FisherInformation, x::Union{Vector{Real},mx.NDArray,Matrix{Real}})
    if fim.result == nothing
        fim.result = mx.zeros(size(x))
    else
        fim.result[:] = 0.
    end

    N = size(fim.jacobian)[end]
    n = convert(Int, round(fim.sample * N))
    iter = fim.subsample < 1.0 ? randperm(fim.rng)[1:n] : 1:N

    for i in iter
        J, M = fim.jacobian[i:i], fim.kl_hessian[i:i]
        @mx.inplace fim.result .+= mx.dot(J',mx.dot(M,mx.dot(J,x)))  #J'*M*J*x 
    end

    @mx.inplace fim.result ./= fim.subsample < 1.0? n : N

    return fim.result
end

# Gauss Newton Approximation to hessian: H ~ jj', j = grad(f)
type GaussNewtonHessian <: FactoredMatrix
   jacobian::Union{Matrix{Real},mx.NDArray}
   subsample::Float64
   rng::Union{AbstractRNG,Void}
   result::Union{mx.NDArray,Vector{Real},Matrix{Real},Void}
end

function mult!(gnh::GaussNewtonHessian, x::Union{Vector{Real},Matrix{Real},mx.NDArray})
    # TODO check if can make more memory efficient
    if gnh.result == nothing
        gnh.result = mx.zeros(size(x))
    else
        gnh.result[:] = 0.
    end

    N = size(gnh.jacobian)[end] # last dimension is the only one that can be sliced
    n = convert(Int, round(gnh.sample * N))
    iter = gnh.subsample < 1.0 ? randperm(gnh.rng)[1:n] : 1:N

    for i in iter
        J = gnh.jacobian[i:i]
        @mx.inplace gnh.result .+= mx.dot(J,mx.dot(J',x))  # J'*J*x 
    end

    @mx.inplace gnh.result ./= gnh.subsample < 1.0? n : N

    return gnh.result
end


## Optimization Stuff

# TODO interface that can interact with an exec (with grad arrays)

function conditioned_conjugate_gradient()
    # TODO
    # check condition number stuff cond(A)
end

function conjugate_gradient(A::FactoredMatrix, b::Union{Vector,mx.NDArray}, x::Union{Vector,mx.NDArray}=mx.zeros(size(b)); num_iter::Int=10 ) 
    # num_iter <-> k

    r = b - mult!(A, x) # might be able to reuse b -- never used again within this scope
    p = mx.copy!( mx.zeros(size(r)), r) 

    _a1 = copy!(zeros(Float32,1,1), mx.dot(r',r) )[1]

    for k = 1:num_iter
        Ap = mult!(A,p)
        _a2 = copy!(zeros(Float32,1,1), mx.dot(p', Ap) )[1]
        a = _a1/_a2
        x += a * p
        rp = r - a * Ap
        if k == num_iter # || norm(rp) <= tol # TODO
            break
        end
        _b1 = copy!(zeros(Float32,1,1), mx.dot(rp',rp) )[1]
        beta = _b1/_a1
        @mx.inplace p .*= beta
        @mx.inplace p .+= rp
        r = rp
        _a1 = _b1
    end

    return x

end

# NOTE mxnet.py ndarray.reshape() is supposed to share memory with original array, so that could be super useful
function trust_region(objective::Function,
                        constraint::Function,
                        th::Union{Vector,mx.NDArray},
                        A::FactoredMatrix, # approximate hessian of constraint
                        b::Union{Vector,mx.NDArray}, # gradientof objective fn
                        del::Float64=0.01,
                        x::Union{Vector,mx.NDArray}=mx.zeros(size(b));
                        num_iter::Int=10,
                        shrink_rate::Float64=0.8)
    #TODO memory efficient operations

    objective_old = objective(th)
    # solve for search direction
    s = conjugate_gradient(A, b, x, num_iter=num_iter)

    # compute maximal step size
    beta = sqrt(2 * del / mx.copy!(zeros(Float32,1,1), mx.dot(s', mult!(A, s) ) )[1])
    @mx.inplace s .*= beta
    th_new = th + s

    # line search
    while (constraint(th, th_new) > del) || objective(th_new) < objective_old
        @mx.inplace s .*= shrink_rate
        th_new = th + s
    end

    return th_new
end
