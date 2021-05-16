include("cliff7.jl")



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


# function test1()
#     Random.seed!(1)
#     n = 6
#     state = all_plus(n)
#     test_state(state)
#     for _ in 1:10n
#         cliff = random_clifford(3)
#         clifford_action!(cliff, state, [2,3,5])
#         test_state(state)
#     end
# end
# test1()


# function test2()
#     n = 20
#     state = all_plus(n)
#     for _ in 1:10n
#         i = rand(1:n)
#         j = rand(1:n)
#         if i==j
#             continue
#         end
#         row_sum!(state, i, j)
#         test_state(state)
#     end
# end
# test2()


n = 2
state = all_plus(2)
ob1 = 0, [false, true]
ob2 = 0, [true, false]
ob3 = 0, [true, false, true, false]
r = measurement!(state, ob1, [1], false)
r = measurement!(state, ob2, [1], true)
r = measurement!(state, ob1, [2], true)