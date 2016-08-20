# Random Projections.jl
# auth: Christopher Ho
# affil: none :|
# date: 08/19/2016
# desc: Implementation of random projections to get a feature embedding that preserves distances:
#       implementing this paper: https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

type RandomProjector
    A::SparseMatrixCSC # NOTE column format, doin matrix-vector operations, so do via 
end

"""
Constructs a (wrapped) sparse random projection matrix
@param m::Int size of input space
@param n::Int size of desired embedding space
@param s::Float sparsity measure, > 0. (1?)
"""
function RandomProjector( m::Int, n::Int, s::Float64=3. , rng::AbstractRNG=RandomDevice() )

    p_zero = 1. - 1./s
    val_p = sqrt(s)
    val_n = -val_p

    # initialize zeros
    A = sprand(rng, m, n, p_zero)

    # can be cheeky and precalculate K random numbers where K = 95% confidence interval of a bionomial(m*n,p_zero)
    #   this assumes that it's faster to do that than one at a time
    # for each nonzero element, has equal likelihood of being positive or negative sqrt(s)
    for i in 1:length(A.nzval)
        A.nzval[i] = rand(rng) > 0.5 ? val_p : val_n
    end

    self = new()
    self.A = A

    return self
end

"""
projects a (large) vector into a space that on average preserves pairwise distance
@param rp::RandomProjector the (wrapped) random projection matrix 
@param x::Vector the vector to project
"""
function project( rp::RandomProjector, x::Vector{Real} )
    # TODO figure out if there's a more efficient way of doing this (BLAS? LAPACK?)
    return vec(rp.A'*x)
end
