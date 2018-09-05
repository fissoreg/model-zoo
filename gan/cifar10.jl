using Flux, Flux.Data.CIFAR10
using Flux: train!

using MLDatasets

using Colors
using Images
using Plots

# get RGB matrix from binary array
#img(x) = reshape([RGB(x[pixel]...) for pixel in Iterators.partition(1:length(x), 3)], 32, 32)
img(x) = [RGB(x[i, j, :]...) for i = 1:size(x, 1), j = 1:size(x, 2)]

# number of training samples to use
ns = 50000

# load data and rescale between -1 and +1
X, Y = MLDatasets.CIFAR10.traindata()
X = X[:, :, :, 1:ns]

#X = Float32.(
#      hcat(
#	[
#	  [c(pixel) for c in (red, green, blue)
#                    for pixel in vec(img)
#	  ]
#	  for img in CIFAR10.images()[1:ns]
#	]...
#      )
#    )

X = 2*X - 1
X = Float64.(X)

n_epochs = 100
batch_size = 100
data_dim = size(X, 1)
noise_dim = 100

#= the discriminator is trained over k batches for every
   update to the generator (small k speeds up learning,
   big k improves quality)
=#
k = 1

# TODO: remove this definition when #145 is merged
const EPS = 1e-7
function binary_crossentropy(y_hat, y)
  return -sum(y .* log.(y_hat + EPS) - (1.0 - y) .* log.(1.0 - y_hat + EPS))
end

function UpSampling(x, h = 2, w = 2)
  W, H, C, N = size(x)
  y = Array{typeof(x.data[1]), 4}(W * h, H * w, C, N)

  # this needs to be fast!
  for i = 1:W, j = 1:H
    for k = (i-1)*h+1:i*h, l = (j-1)*w+1:j*w, m = 1:N
      start = CartesianIndex(k, l, 1, m)
      stop = CartesianIndex(k, l, C, m)
      xstart = CartesianIndex(i, j, 1, m)
      xstop = CartesianIndex(i, j, C, m)
      copy!(y, CartesianRange(start, stop), x.data, CartesianRange(xstart, xstop))
    end
  end

  Flux.Tracker.TrackedArray(y)
end

leaky(x) = leakyrelu(x, 0.2)

# defining the generator
G = Chain(
  Dense(noise_dim, 512 * 4 * 4),
  x -> reshape(x, (4, 4, 512, size(x, 2))),
  UpSampling,
  Conv((3, 3), 512 => 256, pad = (1, 1)), #, pad = h, stride = h)F,,
  # TODO: BatchNorm!
  relu,
  UpSampling,
  Conv((3, 3), 256 => 128, pad = (1, 1)), #, pad = h, stride = h),
  # BatchNorm
  relu,
  UpSampling,
  Conv((3, 3), 128 => 3, pad = (1, 1)), #, pad = h, stride = h),
  tanh
)

# defining the discriminator
D = Chain(
  Conv((2, 2), 3 => 64, stride = (2, 2), leaky),
  Conv((2, 2), 64 => 128, stride = (2, 2), leaky),
  Conv((2, 2), 128 => 256, stride = (2, 2), leaky),
  x -> reshape(x, :, size(x, 4)),
  Dense(4096, 1, sigmoid)
)

# defining losses
d_loss(x, y) = binary_crossentropy(D(x), y)
g_loss(x, y) = binary_crossentropy(D(G(x)), y)

# defining optimisers for discriminator and generator separately
d_opt = ADAM(params(D), 1e-5)
g_opt = ADAM(params(G), 1e-5)

for i=1:n_epochs
  # shuffling and batchifying samples
  X = X[:, :, :, shuffle(1:ns)]
  labels = ones(batch_size)
  batches = [(X[:, :, :,i], labels) for i in Iterators.partition(1:ns, batch_size)]

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
    d_current_loss = d_loss(X[:, :, :, shuffle(1:batch_size)], ones(batch_size))
    g_current_loss = g_loss(randn(noise_dim, batch_size), zeros(batch_size))
    println("[Epoch $i Batch $(min(m, j + k - 1))] d_loss: $d_current_loss g_loss: $g_current_loss")

    # generating samples
    samples = G(randn(noise_dim, 100)).data
    samples = (samples + 1) / 2
    fig = vcat([hcat(img.([samples[:, :, :,i] for i=(j-1)*10+1:j*10])...) for j = 1:10]...)

    savefig(plot(fig), "cifar10_epoch$i")
  end

  # generating samples
  samples = G(randn(noise_dim, 100)).data
  samples = (samples + 1) / 2
  fig = vcat([hcat(img.([samples[:, :, :,i] for i=(j-1)*10+1:j*10])...) for j = 1:10]...)

  savefig(plot(fig), "cifar10_epoch$i")
end
