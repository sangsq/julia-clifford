import Base: +, -, *, /, one, iterate, rand, abs, isless, conj
using Random, LinearAlgebra

struct GF{p}<:Number
    num::Int64
end
GF{p}(a::GF{p}) where p = a

# GF(a::Int, ::Val{p}) where p = GF{p}(mod(a, p))

+(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num + b.num, p))
-(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num - b.num, p))
-(a::GF{p}) where p = GF{p}(p-a.num)
*(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num * b.num, p))
/(a::GF{p}, b::GF{p}) where p = GF{p}(mod(a.num * invmod(b.num, p), p))
isless(a::GF{p}, b::GF{p}) where p = isless(a.num, b.num)
abs(a::GF) = a
conj(a::GF) = a
# one(::GF{p}) where p = GF{p}(1)
# zero(::GF{p}) where p = GF{p}(0)
one(::Type{GF{p}}) where p = GF{p}(1)
zero(::Type{GF{p}}) where p = GF{p}(0)

Random.rand(rng::AbstractRNG, ::Random.SamplerType{GF{p}}) where p = GF{p}(rand(rng, 0:p-1))


zp_inner(a, b) = sum(a .* b)
zp_symplectic_inner(a,b) = zp_inner(a[1:2:end], b[2:2:end]) - zp_inner(a[2:2:end], b[1:2:end])

@views function zp_random_symplectic_matrix(T, n)
    mat = rand(T, 2n, 2n)
    for i in 1:2n
        # display(mat)
        while true
            rand!(mat[i, :])
            if iseven(i) && (zp_symplectic_inner(mat[i, :], mat[i-1, :]) != -one(T))
                continue
            end
            for j in 1:(isodd(i) ? i-1 : i-2)
                tmp = zp_symplectic_inner(mat[i, :], mat[j, :])
                if tmp != zero(T)
                    k = isodd(j) ? j+1 : j-1
                    for l in 1:2n
                        if isodd(j)
                            mat[i, l] = mat[i, l] +  tmp * mat[k, l]
                        else
                            mat[i, l] = mat[i, l] -  tmp * mat[k, l]
                        end
                    end
                end
            end
            if all(mat[i, :] .== zero(T))
                continue
            end
            break
        end
    end
    return mat
end

function zp_P_matrix(T, n)
    mat = zeros(T, 2n, 2n)
    for i in 1:n
        mat[2i-1, 2i] = -one(T)
        mat[2i, 2i-1] = one(T)
    end
    return mat
end

# zp_random_symplectic_matrix(GF{3}, 2)