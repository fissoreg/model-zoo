using Flux, Flux.Data.MNIST
using Flux: log_fast, train!, crossentropy
using Images
using Plots
using CuArrays

ns = 1000

# Load data, binarise it
X = float.(float.(hcat(vec.(MNIST.images()[1:ns])...)) .> 0.5)
X = cu(X)

n  = size(X, 2)
n_epochs = 100
batch_size = 100
data_dim = size(X, 1)
noise_dim = 100

const eps = 1e-7
function binary_crossentropy(y, y_hat)
  return -sum(y .* Flux.log_fast.(y_hat + eps) - (1.0 - y) .* Flux.log_fast.(1.0 - y_hat + eps))
end

function batchify(X, label; batch_size = 100)
  n = size(X, 2)
  labels = label(1, batch_size)
  labels = cu(labels)
  X = X[:, shuffle(1:n)]

  ((X[:,i], labels) for i in Iterators.partition(1:n, batch_size))
end

function samplesToImg(samples; c=10, r=10, h=28, w=28)
  f = zeros(r*h,c*w)
  for i=1:r, j=1:c
    f[(i-1)*h+1:i*h,(j-1)*w+1:j*w] = reshape(samples[:,(i-1)*c+j],h,w)
  end
  w_min = minimum(samples)
  w_max = maximum(samples)
  λ = x -> (x-w_min)/(w_max-w_min)
  map!(λ,f,f)
  colorview(Gray,f)
end

# defining the generator
G = Chain(
  Dense(noise_dim, 256, leakyrelu),
  BatchNorm(256, momentum = 0.8),
  Dense(256, 512, leakyrelu),
  BatchNorm(512, momentum = 0.8),
  Dense(512, 1024, leakyrelu),
  BatchNorm(1024, momentum = 0.8),
  Dense(1024, data_dim, tanh)
)

G = mapleaves(cu, G)

# defining the discriminator
D = Chain(
  Dense(data_dim, 512, leakyrelu),
  Dense(512, 256, leakyrelu),
  Dense(256, 1, sigmoid)
)

D = mapleaves(cu, D)

# defining losses
d_loss(x, y) = crossentropy(D(x), y)
g_loss(x, y) = crossentropy(D(G(x)), y) 

# defining optimisers for discriminator and generator separately
d_opt = ADAM(params(D))
g_opt = ADAM(params(G))

cbd = () -> println(d_loss(X[:, shuffle(1:batch_size)], ones(batch_size)),sum(params(D)[1].grad))
cbg = () -> println(g_loss(randn(noise_dim, batch_size), zeros(batch_size)), sum(params(G)[1].grad))

for i=1:n_epochs
  noise = randn(noise_dim, n)
  println("training...")
  # training the discriminator
  train!(d_loss, batchify(X, ones, batch_size=batch_size), d_opt, cb = cbd)
  println("done 1")
  train!(d_loss, batchify(G(noise), zeros, batch_size=batch_size), d_opt, cb = cbd)
  println("done 2")
  noise = randn(noise_dim, n)

  # training the generator
  train!(g_loss, batchify(noise, ones, batch_size=batch_size), g_opt, cb = cbg)

  samples = G(randn(noise_dim, 100)).data
  img = samplesToImg(samples)
  plot(img, title="Epoch $i")
  gui()
end
