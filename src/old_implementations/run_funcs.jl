using Statistics
include("sym_cliff.jl")


function phi_sim(N, n_step, lag, gate, step_size=1)
    
    state = all_plus(N)
    result = []
    
    for _ in 1:lag
        for i in 1:2:N
            gate(state, i, i % N + 1)
        end
        for i in 2:2:N
            gate(state, i, i % N + 1)
        end
    end
    
    for t in 1:n_step
        push!(result, cal_phi(state, true))
        
        for _ in 1:step_size 
            for i in 1:2:N
                gate(state, i, i % N + 1)
            end
            for i in 2:2:N
                gate(state, i, i % N + 1)
            end
        end
    end
    
    return result
end


function ee_sim(N, n_step, lag, gate, step_size, cut_point=nothing)
        
    state = all_plus(N)
    result = []
    
    if cut_point == nothing
        cut_point = div(N,2)
    end
    
    for _ in 1:lag
        for i in 1:2:N
            gate(state, i, i % N + 1)
        end
        for i in 2:2:N
            gate(state, i, i % N + 1)
        end
    end
    
    for t in 1:n_step
        push!(result, pure_state_bipartite_entropy(state, 1:cut_point))
        
        for _ in 1:step_size 
            for i in 1:2:N
                gate(state, i, i % N + 1)
            end
            for i in 2:2:N
                gate(state, i, i % N + 1)
            end
        end
    end
    
    return result
end


function ee_scale_sim(N, A_list, gate, step_size, lag, n_average)

    state = all_plus(N)
    result = zeros(length(A_list))

    for _ in 1:lag
        for i in 1:2:N
            gate(state, i, i % N + 1)
        end
        for i in 2:2:N
            gate(state, i, i % N + 1)
        end
    end

    for e in 1:n_average
        for _ in 1:step_size
            for i in 1:2:N
                gate(state, i, i % N + 1)
            end
            for i in 2:2:N
                gate(state, i, i % N + 1)
            end
        end

        for i in 1:length(A_list)
            result[i] += pure_state_bipartite_entropy(state, 1:A_list[i])
        end
    end
    return result ./ n_average
end



function mutual_info_sim(N, j_list, gate, step_size, lag, n_average)
    state = all_plus(N)
    result = zeros(length(j_list))


    for _ in 1:lag
        for i in 1:2:N
            gate(state, i, i % N + 1)
        end
        for i in 2:2:N
            gate(state, i, i % N + 1)
        end
    end

    for e in 1:n_average
        for _ in 1:step_size
            for i in 1:2:N
                gate(state, i, i % N + 1)
            end
            for i in 2:2:N
                gate(state, i, i % N + 1)
            end
        end

        for i in 1:length(j_list)
            result[i] += two_point_mutual_info(state, 1, j_list[i])
        end
    end
    return result ./ n_average
end



function correlation_square_sim(N, j_list, gate, step_size, lag, n_average, connected, d_i=Z, d_j=Z)
    state = all_plus(N)
    result = zeros(length(j_list))


    for _ in 1:lag
        for i in 1:2:N
            gate(state, i, i % N + 1)
        end
        for i in 2:2:N
            gate(state, i, i % N + 1)
        end
    end

    for e in 1:n_average
        for _ in 1:step_size
            for i in 1:2:N
                gate(state, i, i % N + 1)
            end
            for i in 2:2:N
                gate(state, i, i % N + 1)
            end
        end

        for i in 1:length(j_list)
            result[i] += two_point_correlation_square(state, 1, j_list[i], connected,  d_i, d_j)
        end
    end
    
    return result ./ n_average
end




function ee_sim_2d(shape, n_step, lag, gate, step_size, region_list)
    
    m, n = shape
    N = m * n
    state = all_plus(N)
    result = []
    
    for _ in 1:lag

        for i in 1:m
            for j in 1:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
            for j in 2:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
        end

        for j in 1:n
            for i in 1:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
            for i in 2:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
        end

    end
    
    for t in 1:n_step
        push!(result, [pure_state_bipartite_entropy(state, region) for region in region_list])
        
        for _ in 1:step_size 
            for i in 1:m
                for j in 1:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
                for j in 2:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
            end

            for j in 1:n
                for i in 1:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
                for i in 2:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
            end
        end
    end
    
    return result
end

function phi_sim_2d(shape, n_step, lag, gate, step_size, connected)
    
    m, n = shape
    N = m * n
    state = all_plus(N)
    result = []
    
    for _ in 1:lag

        for i in 1:m
            for j in 1:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
            for j in 2:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
        end

        for j in 1:n
            for i in 1:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
            for i in 2:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
        end

    end
    
    for t in 1:n_step
        push!(result, cal_phi(state, connected))
        
        for _ in 1:step_size 
            for i in 1:m
                for j in 1:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
                for j in 2:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
            end

            for j in 1:n
                for i in 1:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
                for i in 2:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
            end
        end
    end
    
    return result
end


function both_sim_2d_bcc(shape, n_step, lag, gate, step_size, connected, region)
    
    tt = time()
    
    m, n = shape
    N = m * n
    state = all_plus(N)
    result = []
    
    for _ in 1:lag
        for i in 1:2:m
            for j in 1:2:n
                apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
            end
        end
        for i in 2:2:m
            for j in 2:2:n
                apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
            end
        end
    end
    
    for t in 1:n_step
        push!(result, [cal_phi(state, connected), pure_state_bipartite_entropy(state, region)])     
        for _ in 1:step_size 
            for i in 1:2:m
                for j in 1:2:n
                    apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
                end
            end
            for i in 2:2:m
                for j in 2:2:n
                    apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
                end
            end
        end
    end
    
    print("finished one task in ", time() - tt, "s\n")
    return result
end


function both_sim_2d(shape, n_step, lag, gate, step_size, connected, region)
    
    tt = time()
    
    m, n = shape
    N = m * n
    state = all_plus(N)
    result = []
    
    for _ in 1:lag

        for i in 1:m
            for j in 1:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
            for j in 2:2:n
                apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
            end
        end

        for j in 1:n
            for i in 1:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
            for i in 2:2:m
                apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
            end
        end

    end
    
    for t in 1:n_step
        push!(result, [cal_phi(state, connected), pure_state_bipartite_entropy(state, region)])
        
        for _ in 1:step_size 
            for i in 1:m
                for j in 1:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
                for j in 2:2:n
                    apply_gate_2d!(state, gate, shape, (i, j), (i, j % n + 1))
                end
            end

            for j in 1:n
                for i in 1:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
                for i in 2:2:m
                    apply_gate_2d!(state, gate, shape, (i, j), (i % m + 1, j))
                end
            end
        end
    end
    
    print("finished one task in ", time() - tt, "s\n")
    return result
end


function linreg(x,y)
    b = cov(x,y)/cov(x,x)
    a = mean(y) - b * mean(x)
    return a,b
end

function least_square(x, y, basis_list)
    n = length(x)
    m = length(basis_list)
    X = cat([reshape(a, 1, n) for a in basis_list]..., dims=1)
    tmp = X * transpose(X)
    return inv(tmp) * X * reshape(y, n, 1)
end


function num_diff(a, padding=0)
    big = a[3 + 2 * padding : end]
    small = a[1 : end - 2 - 2 * padding]
    diff = big - small
    return diff
end


function num_2nd_diff(a, padding=0)
    big = a[3 + padding * 2 : end]
    middle = a[2 + padding : end - padding - 1]
    small = a[1:end - 2 - 2 * padding]
    return small .+ big .- 2 .* middle
end



function both_sim_2d_bcc(shape, n_step, lag, gate, step_size, connected, region)
    
    tt = time()
    
    m, n = shape
    N = m * n
    state = all_plus(N)
    result = []
    
    for _ in 1:lag
        for i in 1:2:m
            for j in 1:2:n
                apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
            end
        end
        for i in 2:2:m
            for j in 2:2:n
                apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
            end
        end
    end
    
    for t in 1:n_step
        push!(result, [cal_phi(state, connected), pure_state_bipartite_entropy(state, region)])     
        for _ in 1:step_size 
            for i in 1:2:m
                for j in 1:2:n
                    apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
                end
            end
            for i in 2:2:m
                for j in 2:2:n
                    apply_gate_2d!(state, gate, shape, [(i, j), (i, j % n + 1), (i % n + 1, j), (i % n + 1, j % n + 1)])
                end
            end
        end
    end
    
    print("finished one task in ", time() - tt, "s\n")
    return result
end