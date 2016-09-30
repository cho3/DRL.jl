include(joinpath("..", "src", "DRL.jl"))
using DRL
importall POMDPModels

# stuff to make things work
importall POMDPs
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions

ip = InvertedPendulum()

dqn = rl.DQN(max_steps=1000, checkpoint_interval=25, num_epochs=1000, target_refresh_interval=1000)

rl.solve(dqn, ip)
