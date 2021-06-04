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
        row_sum!(state, i, j)
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
            row_sum!(state1, i, j)
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


