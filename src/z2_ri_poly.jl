import Base: +, -, *, /, one, iterate, rand, abs, isless, conj, iszero, isone, zero, one, isequal, hash, show
using Combinatorics

mutable struct Z2_rip
    poly::Array{Bool, 1}
end


degree(p::Z2_rip) = length(p.poly)-1
iszero(p::Z2_rip) = p.poly==[false]
isone(p::Z2_rip) = p.poly==[true]
one(RI_poly) = RI_poly(Bool[1])
zero(RI_poly) = RI_poly(Bool[0])
isequal(p1::Z2_rip, p2::Z2_rip) = isequal(p1.poly, p2.poly)
hash(p::Z2_rip) = hash(p.poly)
show(io::IO, p::Z2_rip) = begin
    # print("|")
    for i in p.poly
        print(io, i ? "1" : "0")
    end
    # print(")")
end


function prune(a::Array{Bool, 1})
    if !any(a)
        return Bool[0]
    else
        return a[1:findlast(a)]
    end
end


function *(p1::Z2_rip, p2::Z2_rip)
    if iszero(p1) || iszero(p2)
        return Z2_rip(Bool[0])
    end
    n1, n2 = degree(p1), degree(p2)
    tmp = zeros(Bool, n1+n2+1)
    for i in -n1 : n1
        for j in -n2 : n2
            if i + j >= 0
                tmp[i+j+1] ⊻= p1.poly[abs(i)+1] * p2.poly[abs(j)+1]
            end
        end
    end
    return Z2_rip(tmp)
end


function +(p1::Z2_rip, p2::Z2_rip)
    n = max(degree(p1), degree(p2))
    tmp = zeros(Bool, n+1)
    for i in 1:degree(p1)+1
        tmp[i] ⊻= p1.poly[i]
    end
    for i in 1:degree(p2)+1
        tmp[i] ⊻= p2.poly[i]
    end
    return Z2_rip(prune(tmp))
end


-(p1::Z2_rip, p2::Z2_rip) = +(p1, p2)
-(p::Z2_rip) = p


function cqca_g(a)
    return Z2_rip[one(Z2_rip) zero(Z2_rip); Z2_rip(a) one(Z2_rip)]
end


cqca_f() = Z2_rip[zero(Z2_rip) one(Z2_rip); one(Z2_rip) zero(Z2_rip)]
ddet(a::Array{Z2_rip, 2}) = a[1,1]*a[2,2]-a[2,1]*a[1,2]



function ri_completion(a)
    n = length(a)
    tmp = zeros(Bool, 2n-1)
    tmp[1:n] = a[end:-1:1]
    tmp[n:2n-1] = a
    return tmp
end


function cqca_to_bmat(cqca, n)
    @assert size(cqca) == (2, 2)    
    mat = zeros(Bool, 2n, 2n)

    for a in 1:2, b in 1:2
        p = ri_completion(cqca[a, b].poly)
        d = degree(cqca[a, b])
        for i in 1:n
            indices = [mod(i+j-1, n)+1 for j in -d:d]
            for (k, idx) in enumerate(indices)
                mat[2i-2+b, 2idx-2+a] ⊻= p[k]
            end
        end
    end
    return mat
end


function sample_cqca(depth, l, max_degree)
    while true
        s = Z2_rip[one(Z2_rip) zero(Z2_rip); zero(Z2_rip) one(Z2_rip)]
        for i in 1:depth
            s *= cqca_g(prune(rand(Bool, l))) * cqca_f()
        end
        d = maximum(degree.(s))
        if d<= max_degree
            return s
        end
    end
end


function sample_good_cqca(depth, l)
    while true
        s = Z2_rip[one(Z2_rip) zero(Z2_rip); zero(Z2_rip) one(Z2_rip)]
        for i in 1:depth
            s *= cqca_g(prune(rand(Bool, l))) * cqca_f()
        end
        d = maximum(degree.(s))
        if d<= max_degree
            return s
        end
    end
end

function all_barray(n)
    if n==1
        return [[false], [true]]
    else
        tmp = Array{Bool, 1}[]
        for a in all_barray(n-1)
            push!(tmp, [false, a...])
            push!(tmp, [true, a...])
        end
        return tmp
    end
end
        


function get_factor_table(max_d)
    all_polys = [Z2_rip([false, true]), Z2_rip([true, true])]
    for d in 2:max_d
        for p in all_polys[2(2^(d-2)-1)+1:2(2^(d-2)-1)+2^(d-1)]
            push!(all_polys, Z2_rip(Bool[0, p.poly...]))
            push!(all_polys, Z2_rip(Bool[1, p.poly...]))
        end
    end
    # @show all_polys
    ftable = Dict{Z2_rip, Array{Z2_rip, 1}}()
    for p in all_polys
        ftable[p] = [p]
    end
    # @show ftable.keys
    for d in 2:max_d
        for d1 in 1:d-1
            d2 = d-d1
            for p1 in all_polys[2(2^(d1-1)-1)+1:2(2^(d1-1)-1)+2^d1]
                for p2 in all_polys[2(2^(d2-1)-1)+1:2(2^(d2-1)-1)+2^d2]
                    q = p1 * p2
                    if length(ftable[q])==1
                        ftable[q] = cat(ftable[p1], ftable[p2], dims=1)
                    end
                end
            end
        end
    end
    return all_polys, ftable
end


function get_all_cqca(max_d)
    all_polys, ftable = get_factor_table(max_d)
    cqcas = Array{Z2_rip, 2}[]
    for i in 1:div(length(all_polys), 2)
        s_ad = ftable[all_polys[2i-1]]
        s_bc = ftable[all_polys[2i]]
        # @assert isone(all_polys[2i-1]-all_polys[2a= [j]i])
        tmp = Set{Array{Z2_rip, 2}}()
        for mask_a in all_barray(length(s_ad))
            for mask_b in all_barray(length(s_bc))
                s_a = s_ad[mask_a]
                s_d = s_ad[.!mask_a]
                s_b = s_bc[mask_b]
                s_c = s_bc[.!mask_b]
                a = isempty(s_a) ? one(Z2_rip) : *(one(Z2_rip), s_a...)
                b = isempty(s_b) ? one(Z2_rip) : *(one(Z2_rip), s_b...)
                c = isempty(s_c) ? one(Z2_rip) : *(one(Z2_rip), s_c...)
                d = isempty(s_d) ? one(Z2_rip) : *(one(Z2_rip), s_d...)
                if ([a b; c d] in tmp) || ([a c; b d] in tmp) || ([d b; c a] in tmp) || ([d c; b a] in tmp)
                    continue
                end
                push!(tmp, [a b; c d])
                push!(tmp, [-b -a; d c])
            end
        end
        for cqca in tmp
            push!(cqcas, cqca)
        end
    end
    return cqcas
end

        
