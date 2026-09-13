include("sym_cliff.jl")
using PyPlot

m2s(i) = div(i+1, 2)
s2m(i) = 2i-1

function spin_bmat(arcs)
    n = div(length(arcs), 2)
    mat = zeros(Bool, n, 2n)
    l = 1
    for i in 1:2n
        j = arcs[i]
        (j<i) && continue
        si = m2s(i)
        sj = m2s(j)
        if si==sj
            mat[l, i] = mat[l, j] = true
        else
            if i%2 == 1
                mat[l, i+1] = mat[l, j] = true
            else
                mat[l, i-1] = mat[l, j] = true
            end
            mat[l, 2si+1:2sj-2] .= true
        end
        l += 1
    end
    return mat
end

function rand_arcs(n)
    arcs = zeros(Int, 2n)
    tmp = reshape(randperm(2n), n , 2)
    for i in 1:n
        a, b = tmp[i, 1], tmp[i, 2]
        arcs[a], arcs[b] = b, a
    end
    return tmp, arcs
end


function draw_arcs(tmp)
    parc = matplotlib.patches.Arc
    fg, ax = plt.subplots(1, 1)
    for i in 1:size(tmp, 1)
        a, b = tmp[i, :]
        c = (a+b)/2
        r = abs(a-b)
        arc = parc((c, 0), r, r, angle=0, theta1=0, theta2=pi)
        ax.add_patch(arc)
    end
    return fg, ax
end

n = 4

x1 = 1
x2 = 2
x3 = 3

A = 1:x1
B = (x1+1):x2
C = (x2+1):x3
D = (x3+1):n

tmp, arcs = rand_arcs(n)
bmat = spin_bmat(arcs)
state = to_stablizer_state(bmat)

fg, ax = draw_arcs(tmp)

fg.show()
# @show mutual_neg(state, B, C)
# # @show mutual_neg(state, B, D)

# parc = matplotlib.patches.Arc