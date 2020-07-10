using LinearAlgebra
using Random

function zp_uppertrianglize!(m, p)
    dim1, dim2 = size(m)
    finished_rows = 0
    pivs = []
    non_pivs = []
    
    for col in 1: dim2
        # look for first row with non-zeros value at col
        row = nothing
        for i in (1 + finished_rows) : dim1
            if !iszero(m[i, col])
                row = i
                iv = invmod(m[i, col], p)
                m[i, :] .*= iv
                m[i, :] .%= p
                break
            end
        end

        # if not found, skip this col
        if row === nothing
            push!(non_pivs, col)
            continue
        else
            push!(pivs, col)
        end
        m[row, :], m[finished_rows + 1, :] = m[finished_rows + 1, :], m[row, :]

        tmp_m = view(m, :, col: dim2)
        for i in (finished_rows + 2: dim1)
            if  !iszero(tmp_m[i, 1])
                tmp_m[i, :] -= tmp_m[i, 1] * tmp_m[finished_rows + 1, :]
                tmp_m[i, :] %= p
            end
        end

        finished_rows += 1
    end

    return pivs, non_pivs
end


function zp_uppertrianglize(m, p)
    m = copy(m)
    pivs, non_pivs = zp_uppertrianglize!(m, p)
    return m, pivs, non_pivs
end


function zp_row_echlon!(m, p)
    dim1, dim2 = size(m)
    pivs, non_pivs = zp_uppertrianglize!(m, p)
    rk = size(pivs)[1]
    for row in 1:rk
        col = pivs[row]
        for i in 1:(row - 1)
            if !iszero(m[i, col])
                m[i, col: dim2] -= m[i, col] * m[row, col: dim2]
                m[i, col: dim2] %= p
            end
        end
    end
    return pivs, non_pivs
end


function zp_row_echlon(m, p)
    m = copy(m)
    pivs, non_pivs = zp_row_echlon!(m, p)
    return m, pivs, non_pivs
end


function zp_null_space(m, p)
    dim1, dim2 = size(m)
    m = copy(m)
    pivs, non_pivs = zp_row_echlon!(m, p)
    rk = size(pivs)[1]
    perm = cat(pivs, non_pivs, dims=1)

    inv_perm = [1 for _ in 1:dim2]
    [inv_perm[perm[i]] = i for i in 1:dim2]

    non_piv_m = m[:, non_pivs]
    tmp = cat(non_piv_m[1:rk, :], diagm([true for _ in 1:dim2-rk]), dims=1)
    tmp = tmp[inv_perm, :]
    return tmp
end


function zp_rank(m, p)
    return size(zp_uppertrianglize(m, p)[2])[1]
end

# function binary_bidirectional_gaussian!(mat)
#     m, n = size(mat)
#     pivs, _ = binary_uppertrianglize!(mat)
#     row_finished = [false for _ in 1:m]
#     end_points = zeros(Int, m, 2)
#     for col in n:-1:1
#         good_rows = [row for row in 1:m if mat[row, col] && !row_finished[row]]
#         isempty(good_rows) && continue
#         the_row = good_rows[end]

#         for row in good_rows[1:end-1]
#             mat[row, :] .⊻= mat[the_row, :]
#         end

#         row_finished[the_row] = true
#         end_points[the_row, :] = [pivs[the_row], col]
#     end
#     return end_points
# end

# function binary_bidirectional_gaussian(mat)
#     mat = copy(mat)
#     end_points = binary_bidirectional_gaussian!(mat)
#     return mat, end_points
# end

# function zp_inner(x, y)
#     tmp = x .* y
#     return xor(tmp...)
# end

# function binary_symplectic_inner(x, y)
#     a = x[1:2:end] .* y[2:2:end]
#     b = y[1:2:end] .* x[2:2:end]
#     a = xor(a...)
#     b = xor(b...)
#     return xor(a,b)
# end

# function binary_random_symplectic_matrix(n)
#     b_mat = diagm([true for _ in 1:2n])
#     rand_idx = [true for _ in 1:2n]
#     x_img = zeros(Bool, 2n)
#     z_img = zeros(Bool, 2n)

#     for i in 1:n

#         while true
#             x_img .= false
#             rand!(rand_idx)
#             for j in 2i-1 : 2n
#                 rand_idx[j] && (x_img .⊻= b_mat[:, j])
#             end
#             any(x_img) && break
#         end
        
#         while true
#             z_img .= false
#             rand!(rand_idx)
#             for j in 2i-1 : 2n
#                 rand_idx[j] && (z_img .⊻= b_mat[:, j])
#             end
#             any(x_img) && binary_symplectic_inner(x_img, z_img) && break
#         end

#         if i<n
#             x_bad_idx = [j for j in 2i-1:2n if binary_symplectic_inner(x_img, b_mat[:, j])]
#             piv = x_bad_idx[1]
#             for k in x_bad_idx[2:end]
#                 b_mat[:, k] .⊻= b_mat[:, piv]
#             end
#             b_mat[:, piv]= b_mat[:, 2i-1]

#             z_bad_idx = [j for j in 2i:2n if binary_symplectic_inner(z_img, b_mat[:, j])]     
#             piv = z_bad_idx[1]
#             for k in z_bad_idx[2:end]
#                 b_mat[:, k] .⊻= b_mat[:, piv]
#             end
#             b_mat[:, piv] = b_mat[:, 2i]
#         end

#         b_mat[:, 2i-1], b_mat[:, 2i] = x_img, z_img
#     end
    
#     return b_mat
# end


# function binary_random_orthogonal_matrix(n)
#     b_mat = diagm([true for _ in 1:n])
#     rand_idx = [true for _ in 1:n]
#     x_img = zeros(Bool, n)
#     i_finished = zeros(Bool, n)

#     i = 1
#     while i <= n
#         while true
#             x_img .= false
#             rand!(rand_idx)
#             for j in i:n
#                 rand_idx[j] && (x_img .⊻= b_mat[:, j])
#             end
#             if xor(x_img...)
#                 (i > 1) && (i_finished[i-1] = true)
#                 break
#             else
#                 (i == 1 || i_finished[i-1]) && continue
#                 i = i - 1
#             end
#         end

#         if i<n
#             bad_idx = [j for j in i:n if binary_inner(x_img, b_mat[:, j])]
#             tmp = bad_idx[1]
#             for k in bad_idx[2:end]
#                 b_mat[:, k] .⊻= b_mat[:, tmp]
#             end
#             b_mat[:, tmp] = b_mat[:, i]
#             b_mat[:, i] = x_img
#         end

#         i += 1
#     end

#     return b_mat
# end


# function binary_jordan_wigner_transform!(mat)
#     """
#     Turn an orthogonal binary matrix into an Z2 symplectic one or vise versa
#     """
#     m, n = size(mat)
#     @assert (m == n) && (m % 2 == 0)
#     for row in 1:m
#         current_p = false
#         for col in (n-1):-2:1
#             mat[row, col:col+1] .⊻= current_p
#             current_p ⊻= (mat[row, col] ⊻ mat[row, col+1])
#         end
#     end
#     for col in 1:m
#         current_p = false
#         for row in (n-1):-2:1
#             mat[row:row+1, col] .⊻= current_p
#             current_p ⊻= (mat[row, col] ⊻ mat[row+1, col])
#         end
#     end
#     return mat
# end

