import Base: +, -, *, /, one, iterate, rand, abs, isless, conj
using Random, LinearAlgebra

struct GF{p}<:Number
    num::Int64
end
GF{p}(a::GF{p}) where p = a

# GF(a::Int, ::Val{p}) where p = GF{p}(mod(a, p))

+(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num + b.num, p))
-(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num - b.num, p))
*(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num * b.num, p))
/(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num * invmod(b.num, p), p))
isless(a::GF{p}, b::GF{p}) where p = isless(a.num, b.num)
abs(a::GF) = a
conj(a::GF) = a
one(::GF{p}) where p = GF{p}(1)
zero(::GF{p}) where p = GF{p}(0)

Random.rand(rng::AbstractRNG, ::Random.SamplerType{GF{p}}) where p = GF{p}(rand(rng, 0:p-1))

function main()
    a = rand(GF{7}, 8,8)
    rank(a)
end
main()