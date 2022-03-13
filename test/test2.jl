include("cliff7.jl")
include("channels.jl")

function test_state(state)
    xz, s, n_stab = flat(state)
    m, n = size(state)
    for i in 1:2n
        @assert is_herm(s[i], xz[i, :])
        for j in i:2n
            if !(i in 1:m) && !(i in 1+n:m+n)
                continue
            end
            if !(j in 1:m) && !(j in 1+n:m+n)
                continue
            end
            tmp = binary_symplectic_inner(xz[i,:], xz[j, :])
            if i+n==j
                @assert tmp
            else
                @assert !tmp
            end
        end
    end
end

function same_state(state1, state2)
    xz1, s1, _ = flat(state1)
    m1, n1 = size(state1)
    xz2, s2, _ = flat(state2)
    m2, n2 = size(state2)
    for i in 1 : n1
        tmp = s1[i], xz1[i, :]
        outcome = measurement!(state2, tmp, [i for i in 1:n1])
        if outcome
            @assert false
        end
    end
end


function test1()
    Random.seed!(1)
    n = 6
    state = all_plus(n)
    test_state(state)
    for _ in 1:10n
        cliff = random_clifford(3)
        clifford_action!(cliff, state, [2,3,5])
        test_state(state)
    end
end
# test1()


function test2()
    n = 20
    state = all_plus(n)
    for _ in 1:10n
        i = rand(1:n)
        j = rand(1:n)
        if i==j
            continue
        end
        double_row_sum!(state, i, j)
        test_state(state)
    end
end
# test2()


function test_same_state()
    for _ in 1:1000
        m, n = 8, 10
        state = random_state(n, m)
        state1 = copy(state)
        for _ in 1:10
            i = rand(1:m)
            j = rand(1:m)
            while j==i
                j = rand(1:m)
            end
            double_row_sum!(state1, i, j)
        end
        same_state(state1, state)
    end
end
# test_same_state()

function test_depolarize_meas()
    for _ in 1:1000
        m, n = 8, 10
        state = random_state(n, m)
        state1 = copy(state)
        i = rand(1:n)
        depolarize!(state, i)
        measurement!(state1, (0, Bool[1, 0]), [i], false)
        same_state(state1, state)
    end
end
# test_depolarize_meas()


function test_auto_fill_row()
    # Random.seed!(2)
    for _ in 1:1000
        m, n = 8,10
        state = random_state(n, m)
        k = rand(1:m) + rand(Bool) * n
        erase_row!(state, k)
        row_auto_fill!(state, k)
        test_state(state)
    end
    for _ in 1:1000
        m, n = 10, 10
        state = random_state(n, m)
        k = rand(1:m) + n
        u = state.xz[k, :]
        erase_row!(state, k)
        row_auto_fill!(state, k)
        test_state(state)
        @assert u==state.xz[k, :] || u==state.xz[k, :] .⊻ state.xz[k-n, :]
    end
end
# test_auto_fill_row()

function test_ap_neg()
    for _ in 1:10000
        state = random_state(10, 10)
        a = 1
        b = 6
        l = 3
        tmp1 = ap_negativity(state, a, b, l)
        tmp2 = [mutual_neg(state, a+1:a+x, b+1:b+x) for x in 1:l]
        @assert tmp1 == tmp2
    end
end
# test_ap_neg()

function test_ap_mi()
    for _ in 1:1000
        state = random_state(20, 10)
        a = 4
        b = 11
        l = 3
        tmp1 = ap_mutual_info(state, a, b, l)
        tmp2 = [mutual_info(state, a+1:a+x, b+1:b+x) for x in 1:l]
        @assert tmp1 == tmp2
    end
end
# test_ap_mi()

function test_binary_random_symplectic_matrix()
    for _ in 1:100
        n = rand(1:5)
        mat = binary_random_symplectic_matrix(n)
        for i in 1:n
            for j in 1:i
                if i==j
                    @assert binary_symplectic_inner(mat[2i, :], mat[2j-1, :])
                else
                    @assert !binary_symplectic_inner(mat[2i, :], mat[2j-1, :])
                end
                @assert !binary_symplectic_inner(mat[2i, :], mat[2j, :])
                @assert !binary_symplectic_inner(mat[2i-1, :], mat[2j-1, :])
            end
        end
    end
end
# test_binary_random_symplectic_matrix()

function test_measurement()
    Random.seed!(1)
    for i in 1:100
        n = 8
        m = rand(1:n)
        state = random_state(n, m)
        ob = (0, Bool[0, 1])
        a = measurement!(state, ob, [1])
        ss = copy(state)
        b = measurement!(ss, ob, [1])
        same_state(ss, state)
        @assert a==b

    end
end
# test_measurement()

# function tmp()
#     n = 256
#     n_step = 10000


#     state = all_up(n)
#     for _ in 1:n_step
#         for i in 1:n-3
#             cliff = random_clifford(4)
#             clifford_action!(cliff, state, [i, i+1, i+2, i+3])
#         end
#     end
#     return nothing
# end

# using Profile
# Profile.clear()
# Profile.init(Int(1e7), 0.01)
# @profile tmp()


function test_strange_mi()
    a = 5
    b = 8
    for _ in 1:1000
        m = rand(1:a+2b)
        state = random_state(a+2b, m)
        x = strange_AB_mi(state, a)
        y = [mutual_info(state, b+1:b+a, union(b-i+1:b, a+b+1:a+b+i)) for i in 1:b]
        @assert x==y
    end
end
# test_strange_mi()


function test_localizable_EE()
    n = 20
    m = 20
    A = 4:9
    B = 12:16
    for _ in 1:10000
        state = random_state(n, m)
        tmp1 = localizable_EE(state, A, B)

        ob = (0, Bool[1, 0])
        for i in 1:n
            if !(i in A) && !(i in B)
                measurement!(state, ob, [i])
            end
        end
        tmp2 = mutual_info(state, A, B)
        @assert tmp1 == tmp2
    end
end
test_localizable_EE()




function test_tri_mi()
    n = 12
    m = 8
    A = 3:6
    B = 9:10
    C = setdiff(1:n, A, B)
    for _ in 1:10000
        state = random_state(n, m)
        tmp1 = tri_mi(state, A, B, true)
        tmp2 = entropy(state, A) + entropy(state, B) + entropy(state, C) - entropy(state, union(A, B)) - entropy(state, union(B, C))-entropy(state, union(C, A)) + (n-m)
        @assert tmp1 == tmp2
        tmp1 = tri_mi(state, A, B, false)
        tmp2 = entropy(state, C) - entropy(state, union(B, C))-entropy(state, union(C, A)) + (n-m)
        @assert tmp1 == tmp2
    end
    return nothing
end

# test_tri_mi()


function test_mutual_info()
    n = 12
    m = 8
    A = 3:6
    B = 9:10
    for _ in 1:10000
        state = random_state(n, m)
        tmp1 = mutual_info(state, A, B)
        tmp2 = entropy(state, A) + entropy(state, B) - entropy(state, union(A, B))
        @assert tmp1 == tmp2
    end
    return nothing
end

test_mutual_info()