import Base: +, -, *, /, one, iterate, rand, abs, isless, conj, ==
using Random, LinearAlgebra

struct Zp{p}<:Number
    num::Int64
    Zp{p}(num) where p = new(Int64(mod(num, p)))
end


==(a::Zp{p}, b::Zp{p}) where {p} = (a.num == b.num)
-(a::Zp{p}) where {p} = Zp{p}(p-a.num)
+(a::Zp{p}, b::Zp{p}) where {p} = Zp{p}(mod(a.num + b.num, p))
-(a::Zp{p}, b::Zp{p}) where {p} = Zp{p}(mod(a.num - b.num, p))
*(a::Zp{p}, b::Zp{p}) where {p} = Zp{p}(mod(a.num * b.num, p))
/(a::Zp{p}, b::Zp{p}) where {p} = a * Zp{p}(invmod(b.num, p))

*(a::Int, b::Zp{p}) where {p} = Zp{p}(mod(a * b.num, p))
*(a::Zp{p}, b::Int) where {p} = Zp{p}(mod(a.num * b, p))

abs(a::Zp) = a
conj(a::Zp) = a

one(::Type{Zp{p}}) where {p} = Zp{p}(1)
zero(::Type{Zp{p}}) where {p} = Zp{p}(0)

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Zp{p}}) where {p} = Zp{p}(rand(rng, 0:p-1))


function zp_uppertrianglize!(m)
    dim1, dim2 = size(m)
    finished_rows = 0
    pivs = Int[]
    non_pivs = Int[]
    
    for col in 1: dim2
        # look for first row with non-zeros value at col
        row = 0
        for i in (1 + finished_rows) : dim1
            if !iszero(m[i, col])
                row = i
                break
            end
        end

        # if not found, skip this col
        if row == 0
            push!(non_pivs, col)
            continue
        else
            push!(pivs, col)
        end

        for k in max(col-1, 1):dim2
            m[row, k], m[finished_rows + 1, k] = m[finished_rows + 1, k], m[row, k]
        end

        for i in (finished_rows + 2: dim1)
            if !iszero(m[i, col])
                m[i, col : dim2] -= (@view m[finished_rows + 1, col : dim2]) * (m[i, col] / m[finished_rows + 1, col])
            end
        end

        finished_rows += 1
    end

    return pivs, non_pivs
end

function zp_uppertrianglize(m)
    m = copy(m)
    pivs, non_pivs = zp_uppertrianglize!(m)
    return m, pivs, non_pivs
end

function zp_partial_gaussian!(mat::Matrix{T}, indices) where T
    m, n = size(mat)
    is_piv_row = zeros(T, m)
    pivs = Int[]
    for i in indices
        k = 0
        for j in 1:m
            if !iszero(mat[j, i]) && !is_piv_row[j]
                if k==0
                    k = j
                    is_piv_row[k] = true
                    push!(pivs, i)
                else
                    for l in 1:n
                        mat[j, l] -= mat[k, l] * mat[j, i]
                    end
                end
            end
        end
    end
    return is_piv_row, pivs
end


function zp_rank(m)
    return size(zp_uppertrianglize(m)[2])[1]
end


function zp_bidirectional_gaussian!(mat)
    m, n = size(mat)
    pivs, _ = zp_uppertrianglize!(mat)
    row_finished = [false for _ in 1:m]
    end_points = zeros(Int, m, 2)
    for col in n:-1:1
        good_rows = [row for row in 1:m if !iszero(mat[row, col]) && !row_finished[row]]
        isempty(good_rows) && continue
        the_row = good_rows[end]

        for row in good_rows[1:end-1]
            mat[row, :] -= (@view mat[the_row, :]) * (mat[row, col] / mat[the_row, col])
        end

        row_finished[the_row] = true
        end_points[the_row, :] = [pivs[the_row], col]
    end
    return end_points
end

function zp_bidirectional_gaussian(mat)
    mat = copy(mat)
    end_points = zp_bidirectional_gaussian!(mat)
    return mat, end_points
end

function zp_inner(x, y)
    tmp = x * y
    return sum(tmp)
end


function zp_all_vertical_cut_ranks!(b_mat)
    m, n = size(b_mat)
    pivs, _ = zp_uppertrianglize!(b_mat)
    rks = Int[0 for _ in 1:n]
    j = 1
    r = 0
    for k in pivs
        rks[j:k-1] .= r
        j = k
        r += 1
    end
    rks[j:end] .= length(pivs)
    return rks
end


function zp_all_vertical_cut_ranks(b_mat)
    tmp = copy(b_mat)
    return zp_all_vertical_cut_ranks!(tmp)
end


function zp_symplectic_inner(x::Vector{T}, y::Vector{T}) where T
    @assert length(x) == length(y)
    n = size(x, 1)
    r = zero(T)
    for i in 1:2:n
        r += x[i] * y[i+1] - y[i] * x[i+1]
    end
    return r
end


function zp_random_symplectic_matrix(T, n)
    b_mat = rand(T, 2n, 2n)
    tmp = zero(T)
    for i in 1:2n
        while true
            b_mat[i, :] = rand(T,2n)
            if iseven(i)
                tmp = zp_symplectic_inner(b_mat[i-1, :], b_mat[i, :])
                if iszero(tmp)
                    continue
                end
                b_mat[i, :] ./= tmp
                # @assert isone(zp_symplectic_inner(b_mat[i-1, :], b_mat[i, :]))
            end
            for j in 1:(isodd(i) ? i-1 : i-2)
                tmp = zp_symplectic_inner(b_mat[i, :], b_mat[j, :])
                if !iszero(tmp)
                    k = isodd(j) ? j+1 : j-1
                    s = isodd(j) ? T(1) : T(-1)
                    for l in 1:2n
                        b_mat[i, l] += b_mat[k, l] * tmp *s
                    end
                end
                # @assert iszero(zp_symplectic_inner(b_mat[i, :], b_mat[j, :]))
            end
            if all(iszero.(b_mat[i, :]))
                continue
            end
            break
        end
    end
    return b_mat
end

function zp_random_sign_free_symplectic_matrix(T, n)
    b_mat = zeros(T, 2n, 2n)
    tmp = zero(T)
    for i in 1:2n
        while true
            if isodd(i)
                b_mat[i, 1:2:end] = rand(T, n)
            else
                b_mat[i, 2:2:end] = rand(T, n)
            end

            if iseven(i)
                tmp = zp_symplectic_inner(b_mat[i-1, :], b_mat[i, :])
                if iszero(tmp)
                    continue
                end
                b_mat[i, :] ./= tmp
                # @assert isone(zp_symplectic_inner(b_mat[i-1, :], b_mat[i, :]))
            end
            for j in 1:(isodd(i) ? i-1 : i-2)
                tmp = zp_symplectic_inner(b_mat[i, :], b_mat[j, :])
                if !iszero(tmp)
                    k = isodd(j) ? j+1 : j-1
                    s = isodd(j) ? T(1) : T(-1)
                    for l in 1:2n
                        b_mat[i, l] += b_mat[k, l] * tmp *s
                    end
                end
                # @assert iszero(zp_symplectic_inner(b_mat[i, :], b_mat[j, :]))
            end

            if all(iszero.(b_mat[i, :]))
                continue
            end
            break
        end
    end
    return b_mat
end