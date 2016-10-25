# MXNet.jl has no convenient reshape interface fml

# TODO add preallocated input/output matrices for memory efficiency?
type Reshaper
    squasher::mx.Executor
    desquasher::Vector{mx.Executor}
    shapes::Vector{Tuple}
    slice_indices::Vector{Int}
    index_map::Vector{Int}
    symbol_map::Vector{Symbol}
end

function Reshaper(exec::mx.Executor, ctx::mx.Context=mx.cpu())

    # assumes that you're only going to parameterize the variables with gradients enabled
    shapes = [size(grad) for grad in filter((x)->x!=nothing, exec.grad_arrays)]

    index_map = [idx for (idx, grad) in filter((x)->x[1]!=nothing, enumerate(exec.grad_arrays))]

    symbol_map = [symbol(string("param",k)) for k in 1:length(index_map)]

    slice_indices = Int[]

    curr_idx = 0

    for shape in shapes
        curr_idx += prod(shape)
        push!(slice_indices, curr_idx)
    end
    
    inputs = [mx.Variable(sym) for sym in symbol_map]
    squash_arch = @mx.chain mx.Concat(inputs) => mx.Reshape(shape=(curr_idx,))
    symbols_shapes = [(symbol_map[i], shapes[i],) for i = 1:length(symbol_map)]
    squash_exec = mx.simple_bind(squash_arch, ctx; symbols_shapes...)


    desquashers = mx.Executor[]

    for (i,(sym, shape)) in enumerate(zip(symbol_map, shapes))
        len = slice_indices[i] + 1 - ( (i == 1) ? 1 : slice_indices[i-1]) 
        arch = @mx.chain mx.Variable(sym) => mx.Reshape(shape=shape)
        exec = mx.simple_bind(arch, ctx; sym=>(len,))
        push!(desquashers, exec)
    end

    return Reshaper(
                    squash_exec,
                    desquashers,
                    shapes,
                    slice_indices,
                    index_map,
                    symbol_map
                    )
end


function squash(re::Reshaper, xs::Vector{mx.NDArray})

    # TODO simple size checks and whatnot to make sure things are consistent before doing more expensive things

    for (idx, sym) in zip(re.index_map, re.symbol_map)
        mx.copy!( re.squasher.arg_dict[sym], xs[idx]) # TODO `=`? check
    end

    mx.forward( re.squasher )
    return re.squasher.outputs[1]
end

function desquash(re::Reshaper, x::mx.NDArray)
    # TODO simple size checks for reasons...

    ret = mx.NDArray[]

    for (i,(ub, shape,sym)) in enumerate(zip(re.slice_indices, re.shapes, re.symbol_map))
        lb = i == 1 ? 1 : 1 + re.slice_indices[i-1]
        re.desquasher[i].arg_dict[sym] = x[lb:ub] # TODO check
        mx.forward(re.desquasher[i])
        push!(ret, re.desquasher[i].outputs[1]) # TODO check
    end

    return ret
end

