using MLDatasets, Flux, Statistics
using LinearAlgebra
using Random
Random.seed!(1234);

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)


x1 = reshape(train_data.features, 28, 28, 1, :)
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


Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(::BroadcastedOperator{typeof(exp)}, x, g) = let
    J = diagm(exp.(x))
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

import Base.reshape
reshape(x::GraphNode, ndims::GraphNode) = BroadcastedOperator(reshape, x, ndims)
forward(::BroadcastedOperator{typeof(reshape)}, x, ndims) = reshape(x, ndims)
backward(::BroadcastedOperator{typeof(reshape)}, x, ndims, g) =
    tuple(reshape(g, size(x)))

poprawne = 0
suma = 0
logit_cross_entropy(y_predicted::GraphNode, y::GraphNode) = BroadcastedOperator(logit_cross_entropy, y_predicted, y)
forward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y) =
let
    global suma += 1
    println(argmax(y_predicted), argmax(y))
    if argmax(y_predicted) == argmax(y)
        global poprawne += 1
    end
    println("Accuracy: ", poprawne/suma)
    y_shifted = y_predicted .- maximum(y_predicted)
    shifted_logsumexp = log.(sum(exp.(y_shifted)))
    result = y_shifted .- shifted_logsumexp
    loss = -1 .* sum(y .* result, dims=1)
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
# dense(w::GraphNode, b::GraphNode, x::GraphNode, activation) = BroadcastedOperator(dense, w, b, x, activation)
# forward(::BroadcastedOperator{typeof(dense)}, w, b, x, activation) = let
#     activation((w * x) .+ b)
# end
# backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = let
#     tuple(w' * g, g * x', g)
# end


function flatten(x) return reshape(x, length(x)) end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = let
    h,w,c,n = size(x) # h*w*c, n
    reshape(x, length(x))
end
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = let
    tuple(reshape(g, size(x)))
end

function maxPool(x, kernel_size, cache)
    h, w, c, n = size(x)
    output = zeros(h ÷ 2, w ÷ 2, c, n)
    empty!(cache)
    for n=1:n
        for i = 1:c
            for j = 1:h÷2
                @views for k = 1:w÷2
                    val, ids = findmax( x[2*j-1:2*j, 2*k-1:2*k, i, :])
                    output[j, k, i, n] = val

                    idx, idy = ids[1] + 2 * j - 1 - 1, ids[2] + 2 * k - 1 - 1
                    push!(cache, CartesianIndex(idx, idy, i, n))
                end
            end
        end
    end
    return output
end

function maxPoolB(x, g, kernel_size, cache)
        output = zeros(size(x))
        output[cache] = vcat(g...)
        tuple(output)
    end

maxPool(x::GraphNode, kernel_size:: Any, cache ::Any) = BroadcastedOperator(maxPool, x, kernel_size, cache)
forward(::BroadcastedOperator{typeof(maxPool)}, x, kernel_size, cache ::Any) = maxPool(x, kernel_size, cache)
backward(::BroadcastedOperator{typeof(maxPool)}, x, kernel_size ::Any, cache, g) = maxPoolB(x, g, kernel_size, cache)

abstract type NetworkLayer end


mutable struct Network
    layers
end

function Network(layers...)
    return Network(layers)
end


function conv(I, K, b)
    H, W, C, N = size(I)
    HH, WW, C, F = size(K)
    H_R = 1 + H - HH
    W_R = 1 + W - WW
    out = zeros(H_R, W_R, F, N)
    for n=1:N
        for depth=1:F
            for r=1:H_R
                @views for c=1:W_R
                    out[r, c, depth, n] = sum(I[r:r+HH-1, c:c+WW-1, :, n] .* K[:, :, :, depth]) + b[depth]
                end
            end
        end
    end
    return out
end


# function create_kernels(n_input, n_output, kernel_width, kernel_height)
#     # Inicjalizacja Xaviera
#     squid = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
#     random_vals = ones(n_output, n_input, kernel_width, kernel_height) * squid
#     return Variable(random_vals)
# end

function create_kernels(n_input, n_output, kernel_width, kernel_height)
    # Inicjalizacja Xaviera
    squid = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
    random_vals = randn(kernel_height, kernel_width, n_input, n_output) * squid
    return Variable(random_vals)
end
# #x = randn(5, 5, 1, 1)
# #test_o = reshape(x1[:, :, 1], 1, 1, 28, 28)
# test_n = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21568628 0.53333336 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.6745098 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07058824 0.8862745 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19215687 0.07058824 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.67058825 0.99215686 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.11764706 0.93333334 0.85882354 0.3137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09019608 0.85882354 0.99215686 0.83137256 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.14117648 0.99215686 0.99215686 0.6117647 0.05490196 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25882354 0.99215686 0.99215686 0.5294118 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.36862746 0.99215686 0.99215686 0.41960785 0.003921569 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09411765 0.8352941 0.99215686 0.99215686 0.5176471 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.6039216 0.99215686 0.99215686 0.99215686 0.6039216 0.54509807 0.043137256 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.44705883 0.99215686 0.99215686 0.95686275 0.0627451 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.011764706 0.6666667 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686 0.74509805 0.13725491 0.0 0.0 0.0 0.0 0.0 0.15294118 0.8666667 0.99215686 0.99215686 0.52156866 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.99215686 0.8039216 0.3529412 0.74509805 0.99215686 0.94509804 0.31764707 0.0 0.0 0.0 0.0 0.5803922 0.99215686 0.99215686 0.7647059 0.043137256 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7764706 0.043137256 0.0 0.007843138 0.27450982 0.88235295 0.9411765 0.1764706 0.0 0.0 0.18039216 0.8980392 0.99215686 0.99215686 0.3137255 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7137255 0.0 0.0 0.0 0.0 0.627451 0.99215686 0.7294118 0.0627451 0.0 0.50980395 0.99215686 0.99215686 0.7764706 0.03529412 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49411765 0.99215686 0.99215686 0.96862745 0.16862746 0.0 0.0 0.0 0.42352942 0.99215686 0.99215686 0.3647059 0.0 0.7176471 0.99215686 0.99215686 0.31764707 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.53333336 0.99215686 0.9843137 0.94509804 0.6039216 0.0 0.0 0.0 0.003921569 0.46666667 0.99215686 0.9882353 0.9764706 0.99215686 0.99215686 0.7882353 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6862745 0.88235295 0.3647059 0.0 0.0 0.0 0.0 0.0 0.0 0.09803922 0.5882353 0.99215686 0.99215686 0.99215686 0.98039216 0.30588236 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.101960786 0.6745098 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.105882354 0.73333335 0.9764706 0.8117647 0.7137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6509804 0.99215686 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.2509804 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.9490196 0.21960784 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.96862745 0.7647059 0.15294118 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49803922 0.2509804 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;;]
# # test_o = reshape(test_n, 1, 1, 28, 28)
# # test_k = create_kernels2(1, 6, 3, 3)
# test_k = create_kernels(1, 6, 3, 3).output # 2.5269480020289716
# # #println( maximum(test_k1) == maximum(test_k)) # true
# test_b = zeros(6)
# # # using BenchmarkTools
# # println(maximum(conv2(test_n, test_k, test_b))) # 4.957135629854453
# # println(maximum(conv(test_n, test_k)))
# # println(maximum(conv_o(test_o, test_k1, test_b)))

conv(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(conv, x, w, b)
forward(::BroadcastedOperator{typeof(conv)}, x, w, b) = conv(x, w, b)
backward(::BroadcastedOperator{typeof(conv)}, x, w, b, g) = let 
    #N, F, H_R, W_R = size(g)
    H_R, W_R, F, N = size(g)
    #N, C, H, W = size(x)
    H, W, C, N = size(x)
    #F, C, HH, WW = size(w)
    HH, WW, C, F = size(w)
    dx = zeros(size(x))
    dw = zeros(size(w))
    db = zeros(size(b))
    for n=1:N
        for depth=1:F
            for r=1:H_R
                for c=1:W_R
                    wu = w[:, :, :, depth]
                    gje = g[r, c, depth, n]
                    dx[r:r+HH-1, c:c+WW-1, :, n] += wu .* gje
                    dw[:, :, :, depth] += x[r:r+HH-1, c:c+WW-1, :, n] .* g[r, c, depth, n]
                end
            end
        end
    end
    for depth=1:F
        db[depth] = sum(g[:, :, depth, :])
    end
    return tuple(dx, dw, db)
end


function xavier_init(n_input, n_output)
    return Variable(randn(n_input, n_output) * sqrt(6 / (n_input + n_output)))
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
    cache :: Constant{Vector{CartesianIndex{4}}}
    func :: Function
    aMaxPool(kernel_size) = new(Constant(kernel_size), Constant(CartesianIndex{4}[]), maxPool)
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
            x = layer.func(x, layer.kernel_size, layer.cache)
        else
            x = layer.func(x)
        end
    end
    return argmax(forward!(topological_sort(x)))
end

create_graph(n::Network, x) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x, layer.activation)
        elseif layer.func == conv
            x = layer.func(x, layer.weights, layer.bias)
            x = layer.activation(x)
        elseif layer.func == maxPool
            #layer.cache.output .= Constant(CartesianIndex{4}[])
            x = layer.func(x, layer.kernel_size, layer.cache)
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

function update_weights!(n::Network, batchsize, learning_rate)
    for layer in n.layers
        if layer.func == dense || layer.func == conv
            layer.weights.output .-= learning_rate * layer.weights.cache / batchsize
            layer.bias.output .-= learning_rate * layer.bias.cache / batchsize
            layer.bias.cache .= 0
            layer.weights.cache .= 0
        end
    end
end

function update_cache!(n::Network, order)
    backward!(order)
    for layer in n.layers
        if layer.func == dense || layer.func == conv
            layer.weights.cache .+= layer.weights.gradient
            layer.bias.cache .+= layer.bias.gradient
        end
    end
end

settings = (;
    eta = 1e-2,
    epochs = 3,
    batchsize = 100,
)

function loss_and_accuracy(model, data)
    x_e = reshape(data.features, 28, 28, 1, :)
    yhot_e = Flux.onehotbatch(data.targets, 0:9)
    size = 100
    suma = size
    poprawne = 0
    for i=1:size
        x = Variable(x_e[:, :, :, i:i], name="x") 
        y = yhot_e[:, i]
        result = model(x)
        if result == argmax(y)
            poprawne += 1
        end
    end
    acc = round(100 * poprawne/suma; digits=2)
    return acc
end

#@show loss_and_accuracy(net, test_data);  # accuracy about 10%, before training


@time for epoch=1:1
    batchsize = settings.batchsize
    batch_counter = 0
    for i=1:2:600
        batch_counter += 1
        x_val = Variable(x1[:, :, :, i:i], name="x")
        y1 = Variable(yhot[:, i:i], name="y")
        graph = create_graph(net, x_val)
        full = agrad(logit_cross_entropy, graph, y1)
        forward!(full)
        update_cache!(net, full)
        if batch_counter == batchsize
            update_weights!(net, batchsize, settings.eta)
            batch_counter = 0
        end
    end
    acc = loss_and_accuracy(net, train_data)
    test_acc = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
end
end