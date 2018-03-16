using Flux, Flux.Data.CIFAR10
using Flux: train!
using Colors

using Images
using Plots

img(x) = reshape([RGB(x[pixel]...) for pixel in Iterators.partition(1:length(x), 3)], 32, 32)

# number of training samples to use
ns = 10000

# load data and rescale between -1 and +1
X = Float32.(
      hcat(
	[
	  [c(pixel) for c in (red, green, blue)
                    for pixel in vec(img)
	  ]
	  for img in CIFAR10.images()[1:ns]
	]...
      )
    )

X = 2*X - 1

n_epochs = 100
batch_size = 100
data_dim = size(X, 1)
noise_dim = 100

#= the discriminator is trained over k batches for every
   update to the generator (small k speeds up learning,
   big k improves quality)
=#
k = 10

# TODO: remove this definition when #145 is merged
const EPS = 1e-7
function binary_crossentropy(y_hat, y)
  return -sum(y .* log.(y_hat + EPS) - (1.0 - y) .* log.(1.0 - y_hat + EPS))
end

leaky(x) = leakyrelu(x, 0.2)

# defining the generator
G = Chain(
  Dense(noise_dim, 256, leaky),
  BatchNorm(256, momentum = 0.8),
  Dense(256, 512, leaky),
  BatchNorm(512, momentum = 0.8),
  Dense(512, 1024, leaky),
  BatchNorm(1024, momentum = 0.8),
  Dense(1024, data_dim, tanh)
)

# defining the discriminator
D = Chain(
  Dense(data_dim, 512, leaky),
  Dense(512, 256, leaky),
  Dense(256, 1, sigmoid)
)

# defining losses
d_loss(x, y) = binary_crossentropy(D(x), y)
g_loss(x, y) = binary_crossentropy(D(G(x)), y)

# defining optimisers for discriminator and generator separately
d_opt = SGD(params(D), 1e-5)
g_opt = SGD(params(G), 1e-4)

for i=1:n_epochs
  # shuffling and batchifying samples
  X = X[:, shuffle(1:ns)]
  labels = ones(batch_size)
  batches = [(X[:,i], labels) for i in Iterators.partition(1:ns, batch_size)]

  # number of batches
  m = size(batches, 1)

  for j=1:k:m
    # training the discriminator
    for l=j:min(m, j + k - 1)
      train!(d_loss, [batches[l]], d_opt) #, cb = d_cb)

      noise = randn(noise_dim, batch_size)
      train!((x, y) -> -d_loss(x, y), [(G(noise).data, zeros(batch_size))], d_opt) #, cb = d_cb)
    end

    # training the generator
    noise = randn(noise_dim, batch_size)
    train!(g_loss, [(noise, ones(batch_size))], g_opt) #, cb = g_cb)

    # monitoring losses
    d_current_loss = d_loss(X[:, shuffle(1:batch_size)], ones(batch_size))
    g_current_loss = g_loss(randn(noise_dim, batch_size), zeros(batch_size))
    println("[Epoch $i Batch $(min(m, j + k - 1))] d_loss: $d_current_loss g_loss: $g_current_loss")

    # generating samples
    samples = G(randn(noise_dim, 100)).data
    fig = vcat([hcat(img.([samples[:,i] for i=(j-1)*10+1:j*10])...) for j = 1:10]...)
    savefig(plot(fig), "cifar10_epoch$i")
  end
end
