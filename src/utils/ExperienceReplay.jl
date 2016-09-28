# ExperienceReplay.jl
# stuff.

abstract ReplayMemory

size(::ReplayMemory) = 0
push!(::ReplayMemory, ::Vector{Real}, ::Int, ::Real, ::Vector{Real}, td::Real=1.;
        rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}()) = 0
peek(::ReplayMemory; rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}()) = 0
state(::ReplayMemory, idx::Int) = 0

# TODO ref paper
type UniformMemory <: ReplayMemory
    states::NDArray # giant NDArray for speed--probably not too much fatter in memory
    action_indices::Vector{Int} # which action was taken
    rewards::Vector{Real}
    mem_size::Int
    rng::AbstractRNG
end
# TODO make sure push! is equipped to ignore the potential td term
size(mem::UniformMemory) = size(mem.states, 2) / 2
function push!(mem::UniformMemory, 
                s_vec::Vector{Real},
                a_idx::Int,
                r::Real,
                sp_vec::Vector{Real},
                td::Real=1.;
                rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}() )

    if mem.mem_size * 2 > size(mem.states, 2)
        error("Oh shoot something messed up here")
    end

    # if memory is full
    if mem.mem_size * 2 == size(mem.states, 2)
        replace_idx = 0
        if isnull( rng )
            replace_idx = rand(mem.rng, 1:mem_size)
        else
            replace_idx = rand(rng, 1:mem_size)
        end

        mem.action_indices[replace_idx] = a_idx
        mem.rewards[replace_idx] = r

        mem.states[:, replace_idx, :] = s_vec
        mem.states[:, replace_idx + mem.mem_size] = sp_vec

        return
    end


    mem.mem_size += 1

    mem.action_indices[mem.mem_size] = a_idx
    mem.rewards[mem.mem_size] = r

    mem.states[:, mem.mem_size] = s_vec
    mem.states[:, mem.mem_size + size(mem.states, 2)/2] = sp_vec

end

function peek(mem::UniformMemory; rng::Nullable{AbstractRNG}=Nullable{AbstractRNG}() )

    idx = rand( isnull(rng) ? mem.rng : rng, 1:mem.mem_size)

    return idx, mem.action_indices[idx], mem.rewards[idx], idx + size(mem.states, 2) / 2
end

state(mem::UniformMemory, idx::Int) = mem.states[:,idx]




# TODO ref paper
type PrioritizedMemory <: ReplayMemory
    states::NDArray # giant NDArray for speed--probably not too much fatter in memory
    action_indices::Vector{Int} # which action was taken
    rewards::Vector{Real}
    priorities::Vector{Real}
    mem_size::Int
    rng::AbstractRNG
end

size(mem::PrioritizedMemory) = size(mem.states, 2) / 2
# TODO