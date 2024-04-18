using MLDatasets, Flux, Statistics
using LinearAlgebra

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)


function loader(data)
    dim1, dim2, dim3 = size(data.features)
    x = reshape(data.features, dim1 * dim2, dim3)
    y = data.targets
    #x4dim = reshape(data.features, 28, 28, 1, :) # insert trivial channel dim
    yhot  = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    return x, y, yhot
    #Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

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
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
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
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* isless.(x, 0))


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


function flatten(x) return reshape(x, length(x)) end


poprawne = 0
suma2 = 0
logit_cross_entropy(y_predicted::GraphNode, y::GraphNode) = BroadcastedOperator(logit_cross_entropy, y_predicted, y)
forward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y) =
let
    global suma2 += 1
    if argmax(y_predicted) == argmax(y)
        global poprawne += 1
    end
    #println("Accuracy: ", poprawne/suma2)
    y_shifted = y_predicted .- maximum(y_predicted)
    shifted_logsumexp = log.(sum(exp.(y_shifted)))
    result = y_shifted .- shifted_logsumexp
    loss = -1 .* mean(y .* result)
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
function dense(w, x) return w * x end


function flatten(x) return reshape(x, length(x)) end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))

function conv2d() end


function mean_squared_loss(y, ŷ)
    return sum(Constant(0.5) .* (y .- ŷ) .^ Constant(2))
end


abstract type NetworkLayer end

mutable struct Network
    layers
end

function Network(layers...)
    return Network(layers)
end

function conv(I, K, b)
    N, C, H, W = size(I)
    F, C, HH, WW = size(K)
    P = 0
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
                    window = x[n, :, r:r+HH-1, c:c+WW-1]
                    db[depth] += g[n, depth, r, c]
                    dw[depth, :, :, :] += window .* g[n, depth, r, c]
                    dx[n, :, r:r+HH-1, c:c+WW-1] += w[depth, :, :, :] .* g[n, depth, r, c]
                end
            end
        end
    end
    return tuple(dx, dw, db)
                    #dx[n, :, r:r+HH-1, c:c+WW-1] += w[depth, :, :, :] .* g[n, depth, r, c]
                    
    #             end
    #         end
    #     end
    # end
    # for n=1:N
    #     for depth=1:F
    #         for r=1:H_R
    #             for c=1:W_R
    #                 dw[depth, :, :, :] += x[n, :, r:r+HH-1, c:c+WW-1] .* g[n, depth, r, c]
    #             end
    #         end
    #     end
    # end
    # for depth=1:F
    #     db[depth] = sum(g[:, depth, :, :])
    # end
    # return tuple(dx, dw, db)
end

# kernel = [1 0 -1; 2 0 -2; 1 0 -1]
# I = randn(1, 1, 28, 28)
# k = randn(5, 1, 3, 3)
# a = conv(I, k, randn(6))
# println(size(a))
function create_kernels(n_input, n_output, kernel_width, kernel_height)
    random_vals = randn(n_output, n_input, kernel_width, kernel_height) / 100
    return Variable(random_vals)
end



mutable struct aDense <: NetworkLayer
    num_inputs :: Integer
    num_outputs :: Integer
    weights :: Any
    bias :: Any
    activation :: Function
    func :: Function
    aDense(pair, activation) = new(pair[2], pair[1], Variable(randn(pair[2], pair[1])/10), Variable(randn(pair[2])), activation, dense)
end

mutable struct aFlatten <: NetworkLayer
    func :: Function
    aFlatten() = new(flatten)
end

mutable struct aConv2d <: NetworkLayer
    kernel :: Any
    bias :: Any
    activation :: Function
    stride :: Any
    padding :: Any
    func :: Function
end
aConv(filter_size, pair, activation) = aConv2d(create_kernels(pair[1], pair[2], filter_size[1], filter_size[2]), Variable(zeros(pair[2])), activation, nothing, nothing, conv)
aConv(filter_size, pair, activation, stride) = aConv2d(create_kernels(pair[1], pair[2], filter_size[1], filter_size[2]), Variable(zeros(pair[2])), activation, stride, nothing, conv)
aConv(filter_size, pair, activation, stride, padding) = aConv2d(create_kernels(pair[1], pair[2], filter_size[1], filter_size[2]), Variable(zeros(pair[2])), activation, stride, padding, conv)


net = Network(
    #aConv((3, 3), 1 => 6, relu),
    aFlatten(),
    aDense(784 => 25, relu),
    aDense(25 => 10, identity)
)

(n::Network)(x) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x, layer.activation)
        elseif layer.func == conv
            x = layer.func(x, layer.kernel, layer.bias)
            x = layer.activation(x)
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
#backward!(order)
#return [layer.weights.gradient for layer in network.layers]
# x = Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.07058824, 0.07058824, 0.07058824, 0.49411765, 0.53333336, 0.6862745, 0.101960786, 0.6509804, 1.0, 0.96862745, 0.49803922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11764706, 0.14117648, 0.36862746, 0.6039216, 0.6666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.88235295, 0.6745098, 0.99215686, 0.9490196, 0.7647059, 0.2509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19215687, 0.93333334, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.9843137, 0.3647059, 0.32156864, 0.32156864, 0.21960784, 0.15294118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058824, 0.85882354, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7764706, 0.7137255, 0.96862745, 0.94509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3137255, 0.6117647, 0.41960785, 0.99215686, 0.99215686, 0.8039216, 0.043137256, 0.0, 0.16862746, 0.6039216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05490196, 0.003921569, 0.6039216, 0.99215686, 0.3529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54509807, 0.99215686, 0.74509805, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137256, 0.74509805, 0.99215686, 0.27450982, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13725491, 0.94509804, 0.88235295, 0.627451, 0.42352942, 0.003921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31764707, 0.9411765, 0.99215686, 0.99215686, 0.46666667, 0.09803922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.7294118, 0.99215686, 0.99215686, 0.5882353, 0.105882354, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0627451, 0.3647059, 0.9882353, 0.99215686, 0.73333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9764706, 0.99215686, 0.9764706, 0.2509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18039216, 0.50980395, 0.7176471, 0.99215686, 0.99215686, 0.8117647, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15294118, 0.5803922, 0.8980392, 0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.7137255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09411765, 0.44705883, 0.8666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7882353, 0.30588236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09019608, 0.25882354, 0.8352941, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7764706, 0.31764707, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058824, 0.67058825, 0.85882354, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7647059, 0.3137255, 0.03529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21568628, 0.6745098, 0.8862745, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.95686275, 0.52156866, 0.043137256, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.53333336, 0.99215686, 0.99215686, 0.99215686, 0.83137256, 0.5294118, 0.5176471, 0.0627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name="x")
# result = net(x)
# println(result)
# y = Variable([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], name="y")
# agrad(logit_cross_entropy, result, y)
for i=1:60000
#     #x = Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.07058824, 0.07058824, 0.07058824, 0.49411765, 0.53333336, 0.6862745, 0.101960786, 0.6509804, 1.0, 0.96862745, 0.49803922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11764706, 0.14117648, 0.36862746, 0.6039216, 0.6666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.88235295, 0.6745098, 0.99215686, 0.9490196, 0.7647059, 0.2509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19215687, 0.93333334, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.9843137, 0.3647059, 0.32156864, 0.32156864, 0.21960784, 0.15294118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058824, 0.85882354, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7764706, 0.7137255, 0.96862745, 0.94509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3137255, 0.6117647, 0.41960785, 0.99215686, 0.99215686, 0.8039216, 0.043137256, 0.0, 0.16862746, 0.6039216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05490196, 0.003921569, 0.6039216, 0.99215686, 0.3529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54509807, 0.99215686, 0.74509805, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137256, 0.74509805, 0.99215686, 0.27450982, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13725491, 0.94509804, 0.88235295, 0.627451, 0.42352942, 0.003921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31764707, 0.9411765, 0.99215686, 0.99215686, 0.46666667, 0.09803922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.7294118, 0.99215686, 0.99215686, 0.5882353, 0.105882354, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0627451, 0.3647059, 0.9882353, 0.99215686, 0.73333335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9764706, 0.99215686, 0.9764706, 0.2509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18039216, 0.50980395, 0.7176471, 0.99215686, 0.99215686, 0.8117647, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15294118, 0.5803922, 0.8980392, 0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.7137255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09411765, 0.44705883, 0.8666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7882353, 0.30588236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09019608, 0.25882354, 0.8352941, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7764706, 0.31764707, 0.007843138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07058824, 0.67058825, 0.85882354, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.7647059, 0.3137255, 0.03529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21568628, 0.6745098, 0.8862745, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.95686275, 0.52156866, 0.043137256, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.53333336, 0.99215686, 0.99215686, 0.99215686, 0.83137256, 0.5294118, 0.5176471, 0.0627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name="x")
    #x = Variable([0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21568628 0.53333336 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.6745098 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07058824 0.8862745 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19215687 0.07058824 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.67058825 0.99215686 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.11764706 0.93333334 0.85882354 0.3137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09019608 0.85882354 0.99215686 0.83137256 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.14117648 0.99215686 0.99215686 0.6117647 0.05490196 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25882354 0.99215686 0.99215686 0.5294118 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.36862746 0.99215686 0.99215686 0.41960785 0.003921569 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09411765 0.8352941 0.99215686 0.99215686 0.5176471 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.6039216 0.99215686 0.99215686 0.99215686 0.6039216 0.54509807 0.043137256 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.44705883 0.99215686 0.99215686 0.95686275 0.0627451 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.011764706 0.6666667 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686 0.74509805 0.13725491 0.0 0.0 0.0 0.0 0.0 0.15294118 0.8666667 0.99215686 0.99215686 0.52156866 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.99215686 0.8039216 0.3529412 0.74509805 0.99215686 0.94509804 0.31764707 0.0 0.0 0.0 0.0 0.5803922 0.99215686 0.99215686 0.7647059 0.043137256 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7764706 0.043137256 0.0 0.007843138 0.27450982 0.88235295 0.9411765 0.1764706 0.0 0.0 0.18039216 0.8980392 0.99215686 0.99215686 0.3137255 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7137255 0.0 0.0 0.0 0.0 0.627451 0.99215686 0.7294118 0.0627451 0.0 0.50980395 0.99215686 0.99215686 0.7764706 0.03529412 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49411765 0.99215686 0.99215686 0.96862745 0.16862746 0.0 0.0 0.0 0.42352942 0.99215686 0.99215686 0.3647059 0.0 0.7176471 0.99215686 0.99215686 0.31764707 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.53333336 0.99215686 0.9843137 0.94509804 0.6039216 0.0 0.0 0.0 0.003921569 0.46666667 0.99215686 0.9882353 0.9764706 0.99215686 0.99215686 0.7882353 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6862745 0.88235295 0.3647059 0.0 0.0 0.0 0.0 0.0 0.0 0.09803922 0.5882353 0.99215686 0.99215686 0.99215686 0.98039216 0.30588236 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.101960786 0.6745098 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.105882354 0.73333335 0.9764706 0.8117647 0.7137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6509804 0.99215686 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.2509804 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.9490196 0.21960784 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.96862745 0.7647059 0.15294118 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49803922 0.2509804 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
    #y = Variable([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], name="y")
    x = Variable(x1[:, :, i], name="x")
    y = Variable(yhot[:, i], name="y")
    x.output = reshape(x.output, 1, 1, 28, 28)
    graph = net(x)
    order = agrad(logit_cross_entropy, graph, y)
    loss = forward!(order)
    if i % 1000 == 0
        println("Loss: ", loss)
        println("Accuracy: ", poprawne/suma2)
    end
    backward!(order)
    for layer in net.layers
      if hasproperty(layer, :weights)
          layer.weights.output -= 0.01 .* layer.weights.gradient
          layer.bias.output -= 0.01 .* layer.bias.gradient
          #layer.weights.gradient .= 0
          #layer.bias.gradient .= 0
      elseif hasproperty(layer, :kernel)
          layer.kernel.output -= 0.01 .* layer.kernel.gradient
          layer.bias.output -= 0.01 .* layer.bias.gradient
          layer.kernel.gradient .= 0
          layer.bias.gradient .= 0
        
      end
    end
end
