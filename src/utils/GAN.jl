abstract AbstractGAN

type GAN <: AbstractGAN
    generator::NeuralNetwork
    discriminator::NeuralNetwork
    input_dim::Int
    num_epoch::Int
end


function logit(nn::NeuralNetwork)
    y = mx.copy!(mx.zeros(size(get(nn.exec).outputs[1])), get(nn.exec).outputs[1])
    y = 1 + mx.exp(-y)
    z = mx.ones(size(y))
    @mx.inplace z ./= y
    return z
end

glogit(x::mx.NDArray) = (x .* (1 .- x))

function fit!(gan::GAN, data::mx.DataProvider; rng::AbstractRNG=RandomDevice(), num_epoch::Int=20)


    info("Please make sure you don't set an output layer for the discriminator--setup will do that")

    # TODO check if generator output matches discriminator input

    # setup networks
    if isnull(gan.generator.exec)
        initialize!(gan.generator, data)
    end

    if isnull(gan.discriminator.exec)
        initialize!(gan.discriminator, data, held_out_grads=true, need_inpt_grad=true)
    end


    batch_size = mx.get_batch_size(data)
    z = mx.zeros(gan.input_dim, batch_size)
    disc_input_grad = mx.zeros(size(get(gan.generator.exec).outputs[1]))
    # there shouldn't be nothing
    disc_grads = mx.NDArray[mx.zeros(size(g)) for g in get(gan.discriminator.exec).grad_arrays]

    for epoch in 1:gan.num_epoch


        for batch in data
            # generate random numbers (in some range for manifold alignment or whatever
            mx.randn!(0, 1, z)

            # forward on generator
            mx.forward( get(gan.generator.exec); gan.generator.input_name=>z )
            fake_out = get(gan.generator.exec).outputs[1]

            # forward on discriminator
            mx.forward( get(gan.discriminator.exec), is_train=true; gan.discriminator.input_name=>fake_out )

            # compute grad manually
            h = logit( gan.discriminator )
            grad = -glogit(h)

            # backward on discriminator
            mx.backward( get(gan.discriminator.exec), grad )
            # add gradients to disc_grad
            for (g, _g) in zip(disc_grads, get(gan.discriminator.exec).grad_arrays)
                @mx.inplace g .+= _g
            end

            # set gradients for generator: (1/D(x)) * dD/dx ( * dG/dz; G(z) = x )
            grad_g = mx.ones(size(grad)) ./ get(gan.discriminator.exec).outputs[1]
            mx.backward( get(gan.discriminator.exec), grad_g )
            mx.copy!(disc_input_grad, get(gan.discriminator.exec).grad_arrays[1] ) # TODO check

            # forward on discriminator with data
            mx.forward( get(gan.discriminator.exec), is_train=true; gan.discriminator.input_name=>batch )

            # backward on discriminator
            h = logit( gan.discriminator )
            grad = glogit(h)
            mx.backward( get(gan.discriminator.exec), grad )
            # add to disc_grads
            for (g, _g) in zip(disc_grads, get(gan.discriminator.exec).grad_arrays)
                @mx.inplace g .+= _g
            end

            # update discriminator
            update!(gan.disrciminator, grad_arrays=disc_grad)
            clear!( disc_grad )

            # backward on generator from discriminator grads
            mx.backward( get(gan.generator.exec), disc_input_grad )

            # update generator
            update!(gan.generator)

            # update metrics TODO
        end

        # print metrics TODO

        # save model TODO

    end


    return gan
end


type InfoGAN <: AbstractGAN
    generator::NeuralNetwork
    discriminator::NeuralNetwork
    posterior::NeuralNetwork
    # TODO stuff to determin the 'c' stuff
    input_dims::Int # just random noise
    num_epoch::Int
end

# NOTE lazy implementation with three networks rather than 2+1 networks (+1 has shared deep layerS)
function fit!(g::InfoGAN, data::mx.AbstractDataProvider, rng::AbstractRNG=RandomDevice())

    # TODO setup networks...
    if isnull(g.generator.exec)

    end

    if isnull(g.discriminator.exec)

    end

    if isnull(g.posterior.exec)

    end


    for epoch in 1:g.num_epoch

        for batch in data # TODO

            # generate random noise

            # sample from 'c' distributions

            # generate fake samples

            # estimate posterior

            # update posterior

            ## GAN update...
            # get gradients from discrinator on fake

            # update generator

            # get gradients from discriminator on real

            # updater discriminator

        end

        # print statistics TODO

        # checkpoint model TODO


    end


end
