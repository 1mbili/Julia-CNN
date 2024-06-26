using MLDatasets, Flux, Statistics
using LinearAlgebra
using Random
Random.seed!(1234); #Żeby łatwiej testować

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)


dim1, dim2, dim3 = size(train_data.features)
x1 = train_data.features
yhot = Flux.onehotbatch(train_data.targets, 0:9)

abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    cache :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, zero(copy(output)), name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end
import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end
function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end
reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end
update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end
import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)
import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))
import Base: exp
exp(x::GraphNode) = ScalarOperator(exp, x)
forward(::ScalarOperator{typeof(exp)}, x) = return exp(x)
backward(::ScalarOperator{typeof(exp)}, x, g) = tuple(g * exp(x))
import Base: max
max(x::GraphNode) = ScalarOperator(max, x)
forward(::ScalarOperator{typeof(max)}, x) = return max(x)
backward(::ScalarOperator{typeof(max)}, x, g) = tuple(g * exp(x))


import Base: *
import LinearAlgebra: mul!, diagm
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication)pu
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    ones_vec = ones(length(node.output)) # I wektor jednostkowy
    Jx = diagm(y .* ones_vec) # I(length(node.output)) * yI
    Jy = diagm(x .* ones_vec)
    tuple(Jx' * g, Jy' * g)
end
Base.Broadcast.broadcasted(-, x::GraphNode) = BroadcastedOperator(-, x)
forward(::BroadcastedOperator{typeof(-)}, x) = return .- x
backward(::BroadcastedOperator{typeof(-)}, x, g) = tuple(g,-g)


Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)
import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end


Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end


Base.Broadcast.broadcasted(max, x::GraphNode) = BroadcastedOperator(max, x)
forward(::BroadcastedOperator{typeof(max)}, x) = return max.(x)
backward(node::BroadcastedOperator{typeof(max)}, x, g) = let
    index = argmax(node.output)
    𝟏 = zeros(length(x))
    𝟏[index] = 1
    J = 𝟏'
    tuple(J * g)
end

import Base: log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(node::BroadcastedOperator{typeof(log)}, x, g) = let
    𝟏 = ones(length(node.output))
    J = diagm(𝟏 ./ x)
    tuple(J' * g)
end


Base.Broadcast.broadcasted(sin, x::GraphNode) = BroadcastedOperator(sin, x)
forward(::BroadcastedOperator{typeof(sin)}, x) = return cos.(x)
backward(::BroadcastedOperator{typeof(sin)}, x, g) = let
    J = diagm(cos.(x))
    tuple(J' * g)
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(::BroadcastedOperator{typeof(exp)}, x, g) = let
    J = diagm(exp.(x))
    tuple(J' * g)
end


σ(x::Real) = one(x) / (one(x) + exp(-x))
σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0 ./ (1.0 .+ exp.(-x))
backward(::BroadcastedOperator{typeof(σ)}, x, g) = let
    J = diagm(1.0 ./ (1.0 .+ exp.(-x))).*(1.0 .- (1.0 ./ (1.0 .+ exp.(-x))))
    tuple(J' * g)
end

relu(x::Real) = max(zero(x), x)
relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(zero(x), x)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0))


import Base: identity
identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = return x
backward(node::BroadcastedOperator{typeof(identity)}, x, g) = let
    tuple(g)
end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    Jx = diagm(y .* x .^ (y .- 1.0))
    Jy = diagm(log.(abs.(x)) .* x .^ y)
    tuple(Jx' * g, Jy' * g)
end

import Base.reshape
reshape(x::GraphNode, ndims::GraphNode) = BroadcastedOperator(reshape, x, ndims)
forward(::BroadcastedOperator{typeof(reshape)}, x, ndims) = reshape(x, ndims)
backward(::BroadcastedOperator{typeof(reshape)}, x, ndims, g) =
    tuple(reshape(g, size(x)))



logit_cross_entropy(y_predicted::GraphNode, y::GraphNode) = BroadcastedOperator(logit_cross_entropy, y_predicted, y)
forward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y) =
let
    y_shifted = y_predicted .- maximum(y_predicted)
    shifted_logsumexp = log.(sum(exp.(y_shifted)))
    result = y_shifted .- shifted_logsumexp
    loss = -1 .* sum(y .* result)
    return loss
end
backward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y, g) =
let
    y_predicted = y_predicted .- maximum(y_predicted)
    y_predicted = exp.(y_predicted) ./ sum(exp.(y_predicted))
    result = (y_predicted - y)
    return tuple(g .* result)
end

function dense(w, b, x, activation) 
    return activation((w * x) .+ b) 
end
function dense(w, x, activation) return activation(w * x) end


function flatten(x) return reshape(x, length(x)) end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

function maxPool(x, kernel_size)
    println(x[1, 1, :, :])
    aga
    N, C, H, W = size(x)
    K_H = kernel_size[1]
    K_W = kernel_size[2]
    W_2 = fld(W - K_W, K_W) + 1
    H_2 = fld(H - K_H ,K_H) + 1
    if H_2 % 2 == 1
        H -= 1
        W -= 1
    end
    out = zeros(N, C, H_2, W_2)
    for n=1:N
        for c=1:C
            for h=1:K_H:H
                for w=1:K_W:W
                    out[n, c, Int.((w+1)/K_W), Int.((h+1)/K_H)] = maximum(x[n, c, h:h+K_H-1,w:w+K_W-1])
                end
            end
        end
    end
    return out
end


function maxPoolB(x, g, kernel_size)
    println(g[1, 1, :, :])
    N, C, H, W = size(x)
    Gn, Gc, Gh, Gw = size(g)
    dx = zeros(N, C, H, W)
    K_H = kernel_size[1]
    K_W = kernel_size[2]
    for n=1:Gn
        for c=1:Gc
            for h=1:Gh
                for w=1:Gw
                    max_val = x[n, c, 1+(h-1)*K_H:h*(K_H), 1+(w-1)*K_W:w*K_W]
                    max_h, max_w = Tuple.(findmax(max_val)[2])
                    max_h += (h-1)*K_H
                    max_w += (w-1)*K_W
                    dx[n,c,max_h,max_w] = g[n,c,h,w]
                end
            end
        end
    end
    return tuple(dx)
end

maxPool(x::GraphNode, kernel_size:: Any) = BroadcastedOperator(maxPool, x, kernel_size)
forward(::BroadcastedOperator{typeof(maxPool)}, x, kernel_size) = maxPool(x, kernel_size)
backward(::BroadcastedOperator{typeof(maxPool)}, x, kernel_size, g) = maxPoolB(x, g, kernel_size)


function conv(I, K, b)
    N, C, H, W = size(I)
    F, C, HH, WW = size(K)
    H_R = 1 + H - HH
    W_R = 1 + W - WW
    out = zeros(N, F, H_R, W_R)
    for n=1:N
        for depth=1:F
            for r=1:H_R
                for c=1:W_R
                    out[n, depth, r, c] = sum(I[n,:, r:r+HH-1, c:c+WW-1] .* K[depth, :, :, :]) + b[depth]
                end
            end
        end
    end
    println(out[1, 1, :, :])
    dupa
    return out
end

conv(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(conv, x, w, b)
forward(::BroadcastedOperator{typeof(conv)}, x, w, b) = conv(x, w, b)
backward(::BroadcastedOperator{typeof(conv)}, x, w, b, g) = let 
    N, F, H_R, W_R = size(g)
    N, C, H, W = size(x)
    F, C, HH, WW = size(w)
    dx = zeros(size(x))
    dw = zeros(size(w))
    db = zeros(size(b))
    for n=1:N
        for depth=1:F
            for r=1:H_R
                for c=1:W_R
                    wu = w[depth, :, :, :]
                    gje = g[n, depth, r, c]
                    dx[n, :, r:r+HH-1, c:c+WW-1] += wu .* gje
                    dw[depth, :, :, :] += x[n, :, r:r+HH-1, c:c+WW-1] .* g[n, depth, r, c]
                end
            end
        end
    end
    for depth=1:F
        db[depth] = sum(g[:, depth, :, :])
    end
    return tuple(dx, dw, db)
end

function create_kernels(n_input, n_output, kernel_width, kernel_height)
    # Inicjalizacja Xaviera
    squid = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
    random_vals = randn(n_output, n_input, kernel_width, kernel_height) * squid
    return Variable(random_vals)
end

function xavier_init(n_input, n_output)
    return Variable(randn(n_input, n_output) * sqrt(6 / (n_input + n_output)))
end

# Żebym mógł ładnie sieć zdefiniować
abstract type NetworkLayer end
mutable struct Network
    layers
end
function Network(layers...)
    return Network(layers)
end


mutable struct aDense <: NetworkLayer
    weights :: Variable
    bias :: Variable
    activation :: Function
    func :: Function
    aDense(pair, activation) = new(xavier_init(pair[2], pair[1]), Variable(zeros(pair[2])), activation, dense)
end

mutable struct aMaxPool <: NetworkLayer
    kernel_size :: Any
    func :: Function
    aMaxPool(kernel_size) = new(Constant(kernel_size), maxPool)
end

mutable struct aFlatten <: NetworkLayer
    func :: Function
    aFlatten() = new(flatten)
end

mutable struct aConv2d <: NetworkLayer
    weights :: Variable
    bias :: Variable
    activation :: Function
    func :: Function
end
aConv(filter_size, pair, activation) = aConv2d(create_kernels(pair[1], pair[2], filter_size[1], filter_size[2]), Variable(zeros(pair[2])), activation, conv)


net = Network(
    aConv((3, 3), 1 => 6, relu),
    aMaxPool((2,2)),
    aConv((3, 3), 6 => 16, relu),
    aMaxPool((2,2)),
    aFlatten(),
    aDense(400 => 84, relu),
    aDense(84 => 10, identity)
)

(n::Network)(x) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x, layer.activation)
        elseif layer.func == conv
            x = layer.func(x, layer.weights, layer.bias)
            x = layer.activation(x)
        elseif layer.func == maxPool
             x = layer.func(x, layer.kernel_size)
        else
            x = layer.func(x)
        end
    end
    return argmax(forward!(topological_sort(x)))-1
end

create_graph(n::Network, x) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x, layer.activation)
        elseif layer.func == conv
            x = layer.func(x, layer.weights, layer.bias)
            x = layer.activation(x)
        elseif layer.func == maxPool
             x = layer.func(x, layer.kernel_size)
        else
            x = layer.func(x)
        end
    end
    return x
end

agrad(loss_func, y_predicted, y_true) = begin
    loss = loss_func(y_predicted, y_true)
    order = topological_sort(loss)
    return order
end

function update_weights!(n::Network, batchsize)
    for layer in n.layers
        if layer.func == dense
            layer.weights.output .-= layer.weights.cache / batchsize
            layer.bias.output .-= layer.bias.cache / batchsize
            layer.bias.cache .= 0
            layer.weights.cache .= 0
        elseif layer.func == conv
            layer.weights.output .-= layer.weights.cache / batchsize
            layer.bias.output .-= layer.bias.cache / batchsize
            layer.bias.cache .= 0
            layer.weights.cache .= 0
        end
    end
end

function update_cache!(n::Network, order, learning_rate)
    backward!(order)
    for layer in n.layers
        if layer.func == dense
            layer.weights.cache .+= learning_rate .* layer.weights.gradient
            layer.bias.cache .+= learning_rate .* layer.bias.gradient
        elseif layer.func == conv
            layer.weights.cache .+= learning_rate .* layer.weights.gradient
            layer.bias.cache .+= learning_rate .* layer.bias.gradient
        end
    end
end

settings = (;
    eta = 1e-2,
    epochs = 3,
    batchsize = 100,
)

function loss_and_accuracy(model, data)
    x_e = data.features
    yhot_e = Flux.onehotbatch(data.targets, 0:9)
    size = length(data)
    suma = size
    poprawne = 0
    for i=1:size
        x = Variable(reshape(x_e[:, :, i], 1, 1, 28, 28), name="x") 
        y = yhot_e[:, i]
        result = model(x)
        if result == argmax(y)-1
            poprawne += 1
        end
    end
    acc = round(100 * poprawne/suma; digits=2)
    return acc
end




@time for epoch=1:1
    batchsize = settings.batchsize
    batch_counter = 0
    for i=1:size(train_data.features)[3]
        batch_counter += 1
        x = Variable(reshape(x1[:, :, i], 1, 1, 28, 28), name="x")
        y1 = Variable(yhot[:, i], name="y")
        graph = create_graph(net, x)
        full = agrad(logit_cross_entropy, graph, y1)
        forward!(full)
        update_cache!(net, full, settings.eta)
        if batch_counter == batchsize
            update_weights!(net, batchsize)
            batch_counter = 0
        end
    end
    acc = loss_and_accuracy(net, train_data)
    test_acc = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
end  
