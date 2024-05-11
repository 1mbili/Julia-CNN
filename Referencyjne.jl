using MLDatasets, Flux
train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=1)
    x4dim = reshape(data.features, 28, 28, 1, :) # insert trivial channel dim
    yhot  = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

# net = Chain(
#     Flux.flatten,
#     Dense(784 => 25, relu), 
#     Dense(25 => 10, identity),
# )
     
net = Chain(
  Conv((3, 3), 1 => 6, relu),           # 60 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 6 => 16, relu),          # 880 parameters
  MaxPool((2, 2)),
  Flux.flatten,
  Dense(400 => 84, relu),               # 33_684 parameters
  Dense(84 => 10),                      # 850 parameters
)                   # Total: 8 arrays, 35_474 parameters, 139.773 KiB.

x1, y1 = first(loader(train_data)); # (28×28×1×1 Array{Float32, 3}, 10×1 OneHotMatrix(::Vector{UInt32}))
y1hat = net(x1)
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

using Statistics: mean  # standard library
function loss_and_accuracy(model, data)
    (x,y) = only(loader(data; batchsize=length(data)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end

@show loss_and_accuracy(net, test_data);  # accuracy about 10%, before training

train_log = []
settings = (;
    eta = 1e-2,
    epochs = 3,
    batchsize = 100,
)

opt_state = Flux.setup(Descent(settings.eta), net);
for epoch in 1:settings.epochs
    @time for (x,y) in loader(train_data, batchsize=settings.batchsize)
        grads = Flux.gradient(model -> Flux.logitcrossentropy(model(x), y), net)
        Flux.update!(opt_state, net, grads[1])
    end
    
    loss, acc, _ = loss_and_accuracy(net, train_data)
    test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
    nt = (; epoch, loss, acc, test_loss, test_acc) 
    push!(train_log, nt)
end