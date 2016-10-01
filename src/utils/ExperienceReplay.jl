# ExperienceReplay.jl
# stuff.

abstract ReplayMemory

# I hate julia Vectors sometimes
typealias RealVector Union{Vector{Real}, Vector{Int}, Vector{Float64}, Vector{Float32}}

size(::ReplayMemory) = error("Unimplemented")
push!(::ReplayMemory, ::RealVector, ::Int, ::Real, ::RealVector, td::Real=1.;
        rng::Union{Void,AbstractRNG}=nothing) = error("Unimplemented")
peek(::ReplayMemory; rng::Union{Void,AbstractRNG}=nothing) = error("Unimplemented")
state(::ReplayMemory, idx::Int) = error("Unimplemented")

# TODO ref paper
type UniformMemory <: ReplayMemory
    states::mx.NDArray # giant NDArray for speed--probably not too much fatter in memory
    actions::Union{Vector{Int},mx.NDArray} # which action was taken
    rewards::RealVector
    terminals::Vector{Bool}
    mem_size::Int
    vectorized_actions::Bool
    rng::Nullable{AbstractRNG}
end
function UniformMemory(mdp::MDP; 
                        vectorized_actions::Bool=false,
                        mem_size::Int=256, 
                        rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}())
    s = create_state(mdp)
    s_vec = convert(Vector{Float32}, vec(mdp, s) )

    if vectorized_actions
        actions = mx.zeros( dimensions( actions(mdp) ), mem_size * 2 )
    else
        actions = zeros(Int, mem_size)
    end

    # currently pushes to cpu context (by default...)
    return UniformMemory(
                        mx.zeros(length(s_vec), mem_size * 2),
                        actions,
                        zeros(mem_size),
                        falses(mem_size),
                        0,
                        vectorized_actions,
                        rng
                        )
end
size(mem::UniformMemory) = size(mem.states, 2) / 2
function push!(mem::UniformMemory, 
                s_vec::RealVector,
                a::Union{Int,RealVector},
                r::Real,
                sp_vec::RealVector,
                terminalp::Bool=false,
                td::Real=1.;
                rng::Union{Void,AbstractRNG}=nothing )

    if mem.mem_size * 2 > size(mem.states, 2)
        error("Oh shoot something messed up here")
    end

    # if memory is full
    if mem.mem_size * 2 == size(mem.states, 2)
        replace_idx = 0
        if rng == nothing
            replace_idx = rand(mem.rng, 1:mem.mem_size)
        else
            replace_idx = rand(rng, 1:mem.mem_size)
        end


        if mem.vectorized_actions
            mem.actions[replace_idx:replace_idx] = a_idx
        else
            mem.actions[replace_idx] = a_idx
        end
        mem.rewards[replace_idx] = r
        mem.terminals[replace_idx] = terminalp

        mem.states[replace_idx:replace_idx] = reshape(s_vec, length(s_vec), 1)
        idx2 = replace_idx + mem.mem_size
        mem.states[idx2:idx2] = reshape(sp_vec, length(sp_vec), 1)

        return
    end


    mem.mem_size += 1

    if mem.vectorized_actions
        mem.actions[mem.mem_size:mem.mem_size] = a_idx
    else
        mem.actions[mem.mem_size] = a_idx
    end
    mem.rewards[mem.mem_size] = r
    mem.terminals[mem.mem_size] = terminalp

    mem.states[mem.mem_size:mem.mem_size] = reshape(s_vec, length(s_vec), 1)
    idx2 = mem.mem_size + mem.mem_size
    mem.states[idx2:idx2] = reshape(sp_vec, length(sp_vec), 1)

end

function peek(mem::UniformMemory; rng::Union{Void,AbstractRNG}=nothing )

    idx = rand( rng==nothing ? mem.rng : rng, 1:mem.mem_size)

    return idx, 
            mem.vectorized_actions ? idx : mem.actions[idx], 
            mem.rewards[idx], 
            idx + convert(Int,(size(mem.states, 2) / 2)), 
            mem.terminals[idx]
end

state(mem::UniformMemory, idx::Int) = mem.states[idx:idx]
action(mem::UniformMemory, idx::Int) = mem.actions[idx:idx]



# TODO ref paper
# TODO add a bunch of stuff that is consistent with UniformMemory
type PrioritizedMemory <: ReplayMemory
    states::mx.NDArray # giant NDArray for speed--probably not too much fatter in memory
    action_indices::Vector{Int} # which action was taken
    rewards::RealVector
    priorities::RealVector
    mem_size::Int
    rng::AbstractRNG
end

size(mem::PrioritizedMemory) = size(mem.states, 2) / 2
# TODO
