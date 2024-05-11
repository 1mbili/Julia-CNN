using MLDatasets, Flux, Statistics
using StaticArrays
using LinearAlgebra
using Random
Random.seed!(1234)
train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

function loader(data; batchsize::Int=1)
    x4dim = reshape(data.features, 28, 28, 1, :) # insert trivial channel dim
    yhot  = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

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

function backward!(order::Vector; seed::Float32=Float32(1.0))
    result = last(order)
    result.gradient = seed
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
backward(::BroadcastedOperator{typeof(identity)}, x, g) = return tuple(g)

import Base.reshape
reshape(x::GraphNode, ndims::GraphNode) = BroadcastedOperator(reshape, x, ndims)
forward(::BroadcastedOperator{typeof(reshape)}, x, ndims) = reshape(x, ndims)
backward(::BroadcastedOperator{typeof(reshape)}, x, ndims, g) = tuple(reshape(g, size(x)))

logit_cross_entropy(y_predicted::GraphNode, y::GraphNode) = BroadcastedOperator(logit_cross_entropy, y_predicted, y)
forward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y) =
let
    y_shifted = y_predicted .- maximum(y_predicted)
    shifted_logsumexp = log.(sum(exp.(y_shifted)))
    result = y_shifted .- shifted_logsumexp
    loss = -1 .* sum(y .* result, dims=1)
    return tuple(loss)
end
backward(::BroadcastedOperator{typeof(logit_cross_entropy)}, y_predicted, y, g) =
let
    y_predicted = y_predicted .- maximum(y_predicted)
    y_predicted = exp.(y_predicted) ./ sum(exp.(y_predicted), dims=1)
    result = (y_predicted .- y)
    return tuple(g .* result)
end

dense(w::GraphNode, b::GraphNode, x::GraphNode) = BroadcastedOperator(dense, w, b, x)
forward(::BroadcastedOperator{typeof(dense)}, w::Array{Float32, 2}, b::Array{Float32, 1}, x::Matrix{Float32}) = let
    (w * x) .+ b
end
backward(::BroadcastedOperator{typeof(dense)}, w::Array{Float32, 2}, b::Array{Float32, 1}, x::Matrix{Float32}, g::Matrix{Float32}) = let
    dw = g * x'
    dx = w' * g
    db = sum(g, dims=2)
    tuple(dw, db, dx)
end


flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = let
    h, w, c, n = size(x)
    reshape(x, h*w*c, n)
end
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = let
    tuple(reshape(g, size(x)))
end

function maxPool(x::Array{Float32, 4}, kernel_size::Tuple{Int64, Int64}, pool_cache::Vector{CartesianIndex{4}})
    h, w, c, n = size(x)
    kernel_h, kernel_w = kernel_size
    out_h, out_w = div(h, kernel_h), div(w, kernel_w)
    output = Array{Float32}(undef, out_h, out_w, c, n)
    empty!(pool_cache)
    @fastmath @inbounds for m=1:n, i = 1:c ,j = 1:out_h
        j_start = (j - 1) * kernel_h + 1
        j_end = j * kernel_h
        mv_h = kernel_h * j - 2
        @views @inbounds for k = 1:out_w
            k_start = (k - 1) * kernel_w + 1
            k_end = k * kernel_w
            @inbounds window = x[j_start:j_end, k_start:k_end, i, m]
            val, idx_flat = findmax(window)
            @inbounds output[j, k, i, m] = val
            @inbounds idx, idy = idx_flat[1] + mv_h, idx_flat[2] + kernel_w * k - 2
            push!(pool_cache, CartesianIndex(idx, idy, i, m))
        end
    end
    return output
end

function maxPoolB(x::Array{Float32, 4}, g::Array{Float32, 4}, pool_cache:: Vector{CartesianIndex{4}})
    output = zeros(Float32, size(x))
    @fastmath @inbounds for (cache, grad) in zip(pool_cache, g)
        @views output[cache] = grad
    end
    tuple(output)
end

maxPool(x::GraphNode, kernel_size:: Constant{Tuple{Int64, Int64}}, pool_cache:: Constant{Vector{CartesianIndex{4}}}) = BroadcastedOperator(maxPool, x, kernel_size, pool_cache)
forward(::BroadcastedOperator{typeof(maxPool)}, x::Array{Float32, 4}, kernel_size::Tuple{Int64, Int64}, pool_cache::Vector{CartesianIndex{4}}) = maxPool(x, kernel_size, pool_cache)
backward(::BroadcastedOperator{typeof(maxPool)}, x::Array{Float32, 4}, kernel_size::Tuple{Int64, Int64}, pool_cache::Vector{CartesianIndex{4}}, g::Array{Float32, 4}) = maxPoolB(x, g, pool_cache)

abstract type NetworkLayer end

mutable struct Network
    layers
end

function Network(layers...)
    return Network(layers)
end


function im2col(I::Array{Float32, 4}, KH::Int64, KW::Int64, OH::Int64, OW::Int64)
    H, W, C, N = size(I)
    col = Array{Float32}(undef, OH * OW * N, KW * KH * C)
    out_array = Array{Float32}(undef, OH * OW, KW * KH)
    indx = reshape(1:H*W, H,W)[1:H-OH+1,1:W-OW+1]
    for n=1:N,c=1:C
        im = @view I[:, :, c, n]
        @inbounds for (i,value) in enumerate(indx)
          for j = 0:OW-1
            @views out_array[(i-1)*OH*OW+j*OH+1:(i-1)OH*OW+(j+1)OH] = im[value+j*H:value+OH-1+j*H]
          end
        end
        #res = im2col2(im, OH, OW, out_array, indx)
        @views col[(n-1)*OH*OW+1:n*OH*OW, (c-1)*KW*KH+1:c*KW * KH] = out_array
    end
    return col'
  end

function conv(I::Array{Float32, 4}, K::Array{Float32, 4}, b::Array{Float32, 1})
    KH, KW, _, KOC = size(K)
    IH, IW, _, IN = size(I)
    H_O = 1 + IH - KH
    W_O = 1 + IW - KW
    col_I = im2col(I, KH, KW, H_O, W_O)
    col_K = reshape(K, :, KOC)'
    conv = zeros(Float32, KOC, H_O * W_O * IN)
    mul!(conv, col_K, col_I)
    conv .+= b
    output = reshape(conv, KOC, H_O, W_O, IN)
    output = permutedims(output, (2, 3, 1, 4))
    return output
end


function create_kernels(n_input::Int64, n_output::Int64, kernel_width::Int64, kernel_height::Int64)
    # Inicjalizacja Xaviera
    squid::Float32 = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
    random_vals = randn(Float32, kernel_height, kernel_width, n_input, n_output) * squid
    return Variable(random_vals)
end

conv(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(conv, x, w, b)
forward(::BroadcastedOperator{typeof(conv)}, x::Array{Float32, 4}, w::Array{Float32, 4}, b::Array{Float32, 1}) = conv(x, w, b)
backward(::BroadcastedOperator{typeof(conv)}, x::Array{Float32, 4}, w::Array{Float32, 4}, b::Array{Float32, 1}, g::Array{Float32, 4}) = let
    H, W, C, N = size(x)
    HH, WW, _, F = size(w)
    HG, WG, _, _ = size(g)
    H_prime = H - HH + 1
    W_prime = W - WW + 1
    dx = zeros(Float32, H, W, C, N)
    dw = zeros(Float32, HH*WW*C, F)
    db = reshape(sum(g, dims=(1, 2, 4)), F)


    @inbounds for i=1:N
        @views im_col = im2col_back(x[:, :, :, i], HH, WW, H_prime, W_prime, C)
        @views grad_i = g[:, :, :, i]
        #Wagi
        @views grad_col = reshape(grad_i, H_prime*W_prime, F)
        dw .+= im_col * grad_col
        # Input
        @views dim_col = reshape(grad_i, HG*WG, F) * reshape(w, HH*WW*C, F)'
        @views col2im!(dx[:, :, :, i], dim_col, H_prime, W_prime, HH, WW, C)
    end
    dw = reshape(dw, HH, WW, C, F)
    return tuple(dx, dw, db)
end

function im2col_back(x::AbstractArray, HH::Int64, WW::Int64, H_prime::Int64, W_prime::Int64, C::Int64)
    col = zeros(Float32, HH*WW*C, H_prime*W_prime)
    @inbounds for i=1:H_prime, j=1:W_prime
        @inbounds @views col[:, (i-1)*W_prime+j] .= reshape(x[i:i+HH-1, j:j+WW-1, :], HH*WW*C)
    end
    return col
end

function col2im!(dx::AbstractArray, col::AbstractArray, H_prime::Int64, W_prime::Int64, HH::Int64, WW::Int64, C::Int64)
    @inbounds for i=1:H_prime*W_prime-1
        row = @view col[i, :]
        h_index = div(i, W_prime) + 1
        w_index = mod(i, W_prime) + 1
        @inbounds @views dx[h_index:h_index+HH-1, w_index:w_index+WW-1, :] .+= reshape(row, HH, WW, C)
    end
    return dx
end


function xavier_init(n_input::Int64, n_output::Int64)
    val::Float32 = sqrt(6 / (n_input + n_output))
    return Variable(randn(Float32, n_input, n_output) * val)
end


mutable struct aDense <: NetworkLayer
    weights :: Variable
    bias :: Variable
    activation :: Function
    func :: Function
    aDense(pair, activation) = new(xavier_init(pair[2], pair[1]), Variable(zeros(Float32, pair[2])), activation, dense)
end

struct aMaxPool <: NetworkLayer
    kernel_size :: Constant{Tuple{Int64, Int64}}
    cache :: Constant{Vector{CartesianIndex{4}}}
    func :: Function
    aMaxPool(kernel_size) = new(Constant(kernel_size), Constant(CartesianIndex{4}[]), maxPool)
end

struct aFlatten <: NetworkLayer
    func :: Function
    aFlatten() = new(flatten)
end

mutable struct aConv2d <: NetworkLayer
    weights :: Variable
    bias :: Variable
    activation :: Function
    func :: Function
end
aConv(filter_size, pair, activation) = aConv2d(create_kernels(pair[1], pair[2], filter_size[1], filter_size[2]), Variable(zeros(Float32, pair[2])), activation, conv)


net = Network(
    aConv((3, 3), 1 => 6, relu),
    aMaxPool((2,2)),
    aConv((3, 3), 6 => 16, relu),
    aMaxPool((2,2)),
    aFlatten(),
    aDense(400 => 84, relu),
    aDense(84 => 10, identity)
)

(n::Network)(x::Variable) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x)
            x = layer.activation(x)
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

create_graph(n::Network, x::Variable) = begin
    for layer in n.layers
        if layer.func == dense
            x = layer.func(layer.weights, layer.bias, x)
            x = layer.activation(x)
        elseif layer.func == conv
            x = layer.func(x, layer.weights, layer.bias)
            x = layer.activation(x)
        elseif layer.func == maxPool
            x = layer.func(x, layer.kernel_size, layer.cache)
        else
            x = layer.func(x)
        end
    end
    return x
end

agrad(loss_func::Function, y_predicted::BroadcastedOperator{typeof(identity)}, y_true::Variable) = begin
    loss = loss_func(y_predicted, y_true)
    order = topological_sort(loss)
    return order
end

function update_weights!(n::Network, batchsize::Integer, order, learning_rate::Float32)
    backward!(order)
    for layer in n.layers
        if layer.func == dense || layer.func == conv
            layer.weights.output .-= learning_rate .* layer.weights.gradient ./ batchsize
            layer.bias.output .-= learning_rate .* layer.bias.gradient ./ batchsize
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
    size = 10000
    suma = size
    poprawne = 0
    for i=1:size
        x = Variable(x_e[:, :, :, i:i], name="x") 
        y = yhot_e[:, i:i]
        result = model(x)
        if result == argmax(y)
            poprawne += 1
        end
    end
    acc = round(100 * poprawne/suma; digits=2)
    return acc
end
@show loss_and_accuracy(net, test_data);  # accuracy about 10%, before training


x_val = Variable(1, name="x")
y1 = Variable(1, name="y")
graph = create_graph(net, x_val)
full = agrad(logit_cross_entropy, graph, y1)
batchsize = settings.batchsize
for epoch=1:settings.epochs
    @time for (x,y) in loader(train_data, batchsize=batchsize)
        x_val.output = x
        y1.output = y
        forward!(full)
        update_weights!(net, batchsize, full, Float32(0.01))
    end
    acc = loss_and_accuracy(net, train_data)
    test_acc = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
end
