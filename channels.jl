include("cliff7.jl")

"""
if fy != 0, fx == fz == 0
"""
function row_reduce_a_site!(state, i)
    m, n = size(state)
    xz, s, _ = flat(state)
    fx, fy, fz = 0, 0, 0
    for j in 1:m
        if xz[j, 2i-1]==1 && xz[j, 2i]==0
            fx == 0 ? fx = j : row_sum!(state, j, fx)
        elseif xz[j, 2i-1]==0 && xz[j, 2i]==1
            fz == 0 ? fz = j : row_sum!(state, j, fz)
        elseif xz[j, 2i-1]==1 && xz[j, 2i]==1
            fy == 0 ? fy = j : row_sum!(state, j, fy)
        end
    end
    if fy != 0
        if (fx != 0) && (fz != 0)
            row_sum!(state, fy, fx)
            row_sum!(state, fy, fz)
            fy = 0
        elseif (fx != 0) && (fz == 0)
            row_sum!(state, fy, fx)
            fz, fy = fy, 0
        elseif (fx == 0) && (fz != 0)
            row_sum!(state, fy, fz)
            fx, fy = fy, 0
        end
    end
    return fx, fy, fz
end


function depolarize!(state, i)
    fx, fy, fz = row_reduce_a_site!(state, i)
    for tmp in (fx, fy, fz)
        if tmp != 0
            copy_row!(state, tmp, m)
            copy_row!(state, tmp+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
        end
    end
    return nothing
end

function dephase_z!(state, i)
    fx, fy, fz = row_reduce_a_site!(state, i)
    for tmp in (fy, fz)
        if tmp != 0
            copy_row!(state, tmp, m)
            copy_row!(state, tmp+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
        end
    end
    return nothing
end

function replace_up!(state, i)
    fx, fy, fz = row_reduce_a_site!(state, i)
    for tmp in (fx, fy, fz)
        if tmp != 0
            copy_row!(state, tmp, m)
            copy_row!(state, tmp+n, m+n)
            erase_row!(state, m)
            erase_row!(state, m+n)
            state.n_stab -= 1
        end
    end
    state.n_stab += 1
    m, n = size(state)
    xz, s, _ = flat(state)
    xz[m, 2i] = true
    row_auto_fill!(state, m+n)
    return nothing
end