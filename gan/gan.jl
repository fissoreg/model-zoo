using Flux, Flux.Data.MNIST
using Flux: log_fast, train!, crossentropy
using Images
using Plots

plot(sin)

ns = 10000

# Load data
X = float.(hcat(vec.(MNIST.images()[1:ns])...))
X = 2X - 1

n = size(X, 2)
n_epochs = 1000
batch_size = 100
data_dim = size(X, 1)
noise_dim = 100

const eps = 1e-7
function binary_crossentropy(y_hat, y)
  return -sum(y .* Flux.log_fast.(y_hat + eps) - (1.0 - y) .* Flux.log_fast.(1.0 - y_hat + eps))
end

function batchify(X, label; batch_size = 100)
  n = size(X, 2)
  labels = label(batch_size)
  X = X[:, shuffle(1:n)]

  [(X[:,i], labels) for i in Iterators.partition(1:n, batch_size)]
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
d_opt = ADAM(params(D), 1e-4)
g_opt = ADAM(params(G), 1e-4)

cbd = () -> println(d_loss(X[:, shuffle(1:batch_size)], ones(batch_size)),sum(params(D)[1].grad))
cbg = () -> println(g_loss(randn(noise_dim, batch_size), zeros(batch_size)), sum(params(G)[1].grad))

k = 10

for i=1:n_epochs

  data = batchify(X, ones, batch_size=batch_size)
  #generated = batchify(G(randn(noise_dim, size(X,2))), zeros, batch_size=batch_size)

  # # of batches
  m = size(data, 1)

  println("Epoch $i")

  for j=1:k:m
    stop = j + k > m ? m : j + k
    for l=j:stop
      # training the discriminator
      train!(d_loss, [data[l]], d_opt) #, cb = cbd)
      dd = D(X)
      dd_min = minimum(dd.data)
      #println("Grad: ", sum(params(D)[1].grad))
      train!(d_loss, batchify(G(randn(noise_dim, batch_size)).data, zeros, batch_size=batch_size), d_opt) #, cb = cbd)
      noise = randn(noise_dim, 1000)
      gdata = G(noise).data
      dg = D(gdata)
      dg_min = minimum(dg.data)
      #println("Grad: ", sum(params(D)[1].grad))
      println("Epoch $i Batch $l\n",
	      "Data: ", dd_min, "\t", d_loss(data[l][1], data[l][2]),
	      "\nGdat: ", dg_min, "\t", d_loss(gdata, zeros(size(gdata, 2))),
	      "\nGen loss: ", g_loss(noise, ones(size(noise, 2)))
	     )
    end

    # training the generator
    noise = randn(noise_dim, batch_size)
    train!(g_loss, batchify(noise, ones, batch_size=batch_size), g_opt) #, cb = cbg)


  end

  samples = G(randn(noise_dim, 100)).data
  img = samplesToImg(samples)
  savefig(plot(img, title="Epoch $i"), "epoch$i.png")
  #gui()
end
