
mx.copy!(dst::mx.NDArray, src::RealVector) = copy!( dst, convert( Array{Float32,2}, reshape(src, length(src), 1) ) )

# because simple_bind offers no way to allow for computation of input gradients
function simple_bind2(self :: mx.SymbolicNode, ctx :: mx.Context;
    grad_req :: Union{mx.GRAD_REQ, Dict{Symbol, mx.GRAD_REQ}}=mx.GRAD_WRITE,
    kwargs...)
    arg_shapes, out_shapes, aux_shapes = mx.infer_shape(self; kwargs...)
    @assert(!isa(arg_shapes, Void), "Information not enough to perform complete shape inference")

    arg_arrays = mx.NDArray[mx.zeros(shape, ctx) for shape in arg_shapes]
    arg_names  = mx.list_arguments(self)

    grad_arrays = Dict{Symbol,mx.NDArray}()

    if grad_req != mx.GRAD_NOP
        shapes = zip(arg_names, arg_shapes)

        # if isa(grad_req, Dict{Symbol, GRAD_REQ})
        #  shapes = filter(x -> grad_req[x[1]] != GRAD_NOP,shapes)
        # end

        for (name, shape) in shapes
            grad_arrays[name] = mx.zeros(shape, ctx)
        end
    end

    aux_arrays = [mx.zeros(shape, ctx) for shape in aux_shapes]
    return mx.bind(self, ctx, arg_arrays, args_grad=grad_arrays, grad_req=grad_req, aux_states=aux_arrays)
end


