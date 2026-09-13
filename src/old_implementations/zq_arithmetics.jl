import Base: +, -, *, /, one, iterate, rand, abs, isless, conj
using Random, LinearAlgebra

struct Zq{p,n}<:Number
    num::Int64
end


-(a::Zq{p, n}) where {p, n} = Zq{p, n}(p^n-a.num)
+(a::Zq{p, n}, b::Zq{p, n}) where {p, n} = Zq{p, n}(mod(a.num + b.num, p^n))
-(a::Zq{p, n}, b::Zq{p, n}) where {p, n} = Zq{p, n}(mod(a.num - b.num, p^n))
*(a::Zq{p, n}, b::Zq{p, n}) where {p, n} = Zq{p, n}(mod(a.num * b.num, p^n))

*(a::Int, b::Zq{p, n}) where {p, n} = Zq{p, n}(mod(a * b.num, p^n))
*(a::Zq{p, n}, b::Int) where {p, n} = Zq{p, n}(mod(a.num * b, p^n))

abs(a::Zq) = a
conj(a::Zq) = a

one(::Type{Zq{p, n}}) where {p, n} = Zq{p, n}(1)
zero(::Type{Zq{p, n}}) where {p, n} = Zq{p, n}(0)

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Zq{p, n}}) where {p, n} = Zq{p, n}(rand(rng, 0:p^n-1))

function order(a::Zq{p, n}) where {p,n}
    num = a.num
    @assert 0 <= num < p^n
    if iszero(num)
        return n
    else
        k = mod(num, p)
        if k == 0
            tmp = div(num, p)
            return 1 + order(Zq{p,n}(tmp))
        else
            return 0
        end
    end
end

function ordinv(a::Zq{p, n}) where {p, n}
    # return the k such that a*k = p^order(a)
    i = order(a)
    if i==n
        @assert false
    end
    tmp = div(a.num, p^i)
    k = invmod(tmp, p^(n-i))
    @assert mod(a.num * k, p^n) == p^i
    return k
end

# function upper_triangularize(mat)
#     m, n = size(mat)
#     for j in 1:n
#         ords = [order(mat[i, j]) for i in 1:m]