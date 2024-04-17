println("a")
using MLDatasets, Flux
println("b")
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

x1, y1, yhot = loader(train_data);
struct Dual{T <:Number} <:Number
    v::T
   dv::T
end
# Przeciążenie podstawowych operatorów
import Base: +, -, *, /
-(x::Dual)          = Dual(-x.v,       -x.dv)
+(x::Dual, y::Dual) = Dual( x.v + y.v,  x.dv + y.dv)
-(x::Dual, y::Dual) = Dual( x.v - y.v,  x.dv - y.dv)
*(x::Dual, y::Dual) = Dual( x.v * y.v,  x.dv * y.v + x.v * y.dv)
/(x::Dual, y::Dual) = Dual( x.v / y.v, (x.dv * y.v - x.v * y.dv)/y.v^2)
# Przeciążenie podstawowych funkcji
import Base: abs, sin, cos, tan, exp, sqrt, isless
abs(x::Dual)  = Dual(abs(x.v),sign(x.v)*x.dv)
sin(x::Dual)  = Dual(sin(x.v), cos(x.v)*x.dv)
cos(x::Dual)  = Dual(cos(x.v),-sin(x.v)*x.dv)
tan(x::Dual)  = Dual(tan(x.v), one(x.v)*x.dv + tan(x.v)^2*x.dv)
exp(x::Dual)  = Dual(exp(x.v), exp(x.v)*x.dv)
sqrt(x::Dual) = Dual(sqrt(x.v),.5/sqrt(x.v) * x.dv)
isless(x::Dual, y::Dual) = x.v < y.v;
# Promocja typów i konwersja
import Base: convert, promote_rule
convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.v), convert(T, x.dv))
convert(::Type{Dual{T}}, x::Number) where T = Dual(convert(T, x), zero(T))
promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}
# Pomocne funkcje
import Base: show
show(io::IO, x::Dual) = print(io, "(", x.v, ") + [", x.dv, "ϵ]");
value(x::Dual) = x.v;
partials(x::Dual) = x.dv;
ReLU(x) = max(zero(x), x)
σ(x) = one(x) / (one(x) + exp(-x))
tanh(x) = 2.0 / (one(x) + exp(-2.0x)) - one(x)
D = derivative(f, x) = partials(f(Dual(x, one(x))))
J = function jacobian(f, args::Vector{T}) where {T <:Number}
    jacobian_columns = Matrix{T}[]
    
    for i=1:length(args)
        x = Dual{T}[]
        for j=1:length(args)
            if i == j
                push!(x, Dual(args[j], one(args[j])))
            else
                push!(x, Dual(args[j], zero(args[j])))
            end
        end
        column = partials.([f(x)...])
        println("Kolumna: ", column)
        push!(jacobian_columns, column[:,:])
    end
    hcat(jacobian_columns...)
end

H = function hessian(f, args::Vector)
    ∇f(x::Vector) = J(f, x)
    J(∇f, args)
end
mean_squared_loss(y::Vector, ŷ::Vector) = sum(0.5(y - ŷ).^2)
fullyconnected(w::Vector, n::Number, m::Number, v::Vector, activation::Function) = activation.(reshape(w, n, m) * v)
n1 = 50
input = 784
output = 10
Wh  = randn(n1,input) # 50 x 784
Wo  = randn(output,n1) # 10 x 50
dWh = similar(Wh)
dWo = similar(Wo)
E = Float64[]

x = x1[:, 1]
y = y1[1]
yhot = yhot[:,1]

function net(x, wh, wo, yhot)
    println(wh[1])
    x̂ = fullyconnected(wh, n1, input, x, σ)
    ŷ = fullyconnected(wo, output, n1, x̂, u -> u)
    println(ŷ, yhot)
    E = Flux.logitcrossentropy(ŷ, yhot)
end

Ei = net(x, Wh[:],  Wo[:], yhot)
println("WH: ", Wh[1])
dnet_Wh(x, wh, wo, yhot) = J(w -> net(x, w, wo, yhot), wh);
dWh[:] = dnet_Wh(x, Wh[:], Wo[:], yhot);
dnet_Wo(x, wh, wo, yhot) = J(w -> net(x, wh, w, yhot), wo);
dWo[:] = dnet_Wo(x, Wh[:], Wo[:], yhot);
for i=1:2
    push!(E, Ei)
    println(Ei)
    Wh .-= 0.04dWh
    Wo .-= 0.04dWo
    Ei  = net(x, Wh[:], Wo[:], yhot)
end

