include("./cliff7.jl")

"""
if fy != 0, then fx == fz == 0
"""
function row_reduce_a_site!(state, i)
    m, n = size(state)
    xz, s, _ = flat(state)
    fx, fy, fz = 0, 0, 0
    for j in 1:m
        if xz[j, 2i-1]==true && xz[j, 2i]==false
            fx == 0 ? fx = j : double_row_sum!(state, j, fx)
        elseif xz[j, 2i-1]==false && xz[j, 2i]==true
            fz == 0 ? fz = j : double_row_sum!(state, j, fz)
        elseif xz[j, 2i-1]==true && xz[j, 2i]==true
            fy == 0 ? fy = j : double_row_sum!(state, j, fy)
        end
    end
    if fy != 0
        if (fx != 0) && (fz != 0)
            double_row_sum!(state, fy, fx)
            double_row_sum!(state, fy, fz)
            fy = 0
        elseif (fx != 0) && (fz == 0)
            double_row_sum!(state, fy, fx)
            fz, fy = fy, 0
        elseif (fx == 0) && (fz != 0)
            double_row_sum!(state, fy, fz)
            fx, fy = fy, 0
        end
    end
    return fx, fy, fz
end


function depolarize!(state, i)
    m, n = size(state)
    fx, fy, fz = row_reduce_a_site!(state, i)
    rows = [fx, fy, fz]
    sort!(rows, rev=true)
    for tmp in rows
        if tmp != 0
            copy_row!(state, tmp, m)
            copy_row!(state, tmp+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
            m -= 1
        end
    end
    return nothing
end


function dephase_z!(state, i)
    m, n = size(state)
    _, fy, fz = row_reduce_a_site!(state, i)
    rows = [fy, fz]
    sort!(rows, rev=true)
    for tmp in rows
        if tmp != 0
            copy_row!(state, tmp, m)
            copy_row!(state, tmp+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
            m -= 1
        end
    end
    return nothing
end


function replace_up!(state, i)
    depolarize!(state, i)
    m, n = size(state)
    state.n_stab += 1
    m += 1
    xz, s, _ = flat(state)
    xz[m, 2i] = true
    row_auto_fill!(state, m+n)
    return nothing
end