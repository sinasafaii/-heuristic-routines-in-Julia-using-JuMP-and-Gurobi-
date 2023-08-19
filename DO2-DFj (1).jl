using IJulia
IJulia.installkernel("DO2", "--history-size=15000")

using GraphsFlows

using JuMP, Gurobi, MathOptInterface
using LinearAlgebra
using Graphs
using GLPK
using TravelingSalesman
using TravelingSalesmanExact

module TravelingSalesman
    include("struct.jl")
    include("util.jl")
    include("generator.jl")
    include("mtz.jl")
    include("tree.jl")
    include("dfj.jl")
end

struct TSPData
    pos::Matrix{Int64}
    cost::Matrix{Int64}
end
struct Solution
    from::Vector{Int64}
    to::Vector{Int64}
    cost::Int64
end

function solve_tsp_dfj(data::TSPData)::Solution
    nnodes = size(data.cost, 1)
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "TimeLimit", 600)
    @variable(m, x[1 : nnodes, 1 : nnodes], Bin)
    @objective(m, Min, dot(data.cost, x))
    @constraint(m, inflow[i in 1 : nnodes], sum(x[:, i]) == 1)
    @constraint(m, outflow[i in 1 : nnodes], sum(x[i, :]) == 1)


    function heuristic_callback(cb_data)
        xval = callback_value.(cb_data, x)
        let 
            sol = modified_rounding_heuristic(data, xval)
            if !isnothing(sol)
                yval = zeros(Int64, nnodes, nnodes)
                for k in 1 : nnodes
                    i, j = sol.from[k], sol.to[k]
                    yval[i, j] = 1
                end
                status = MOI.submit(m, MOI.HeuristicSolution(cb_data), vec(x), vec(yval))
                println("modified rounding heuristic found a solution, status = $status")
            end
        end
    end

    function lazy_callback(cb_data)
        xval = callback_value.(cb_data, x)

        capcut = separate_capcut(data, xval)
        if !isnothing(capcut) && !isempty(capcut)
            println("Capacity cut found = $capcut")
            con = @build_constraint(sum(x[i, j] for i in capcut, j in 1 : nnodes if !in(j, capcut)) >= 1)
            MOI.submit(m, MOI.LazyConstraint(cb_data), con)
        end        
        flowcut =max_flow_min_cut(data, xval)
        if !isnothing(flowcut) && !isempty(flowcut)
            #println("Flow cut found = $flowcut")
            con = @build_constraint(sum(x[i, j] for i in flowcut, j in 1 : nnodes if !in(j, flowcut)) >= 1)
            MOI.submit(m, MOI.LazyConstraint(cb_data), con)
        end        
        
    end

MOI.set(m, MOI.HeuristicCallback(), heuristic_callback)
MOI.set(m, MOI.LazyConstraintCallback(), lazy_callback)

    optimize!(m)

    from = Int64[]
    to = Int64[]
    cost = 0
    if termination_status(m) == MOI.OPTIMAL
        xval = round.(Int64, value.(x))
        for i in 1 : nnodes, j in 1 : nnodes
            if xval[i, j] == 1
                push!(from, i)
                push!(to, j)
                cost += data.cost[i, j]
            end
        end
    end
    return Solution(from, to, cost)
end

function separate_capcut(data::TSPData, x)::Union{Nothing, Set{Int64}}
    nnodes = size(x, 1)
    y =x
    # compute the connected components induced by y
    g = Graphs.SimpleGraph(nnodes)
    for i in 1 : nnodes, j in i + 1 : nnodes
        if y[i, j] + y[j, i] >= 1e-5
            add_edge!(g, i, j)
        end
    end
    connected_comps = connected_components(g)
    if length(connected_comps) <= 1 #graph is connected, found nothing!
        return nothing 
    else # return the cut of the first connected component
        return Set(connected_comps[1])
    end
end

function modified_rounding_heuristic(data::TSPData, x)::Union{Nothing, Solution}
    nnodes = size(x, 1)
    y = zeros(nnodes, nnodes)
    for i in 1:nnodes
        for j in 1:nnodes
            if x[i,j] >= 0.25
                x[i,j] = 1
            else
                x[i,j] = 0
            end
        end
    end

    # check that y forms a connected graph
    g = Graphs.SimpleGraph(nnodes)
    for i in 1 : nnodes, j in i + 1 : nnodes
        if y[i, j] + y[j, i] >= 1
            add_edge!(g, i, j)
        end
    end
    if !is_connected(g) # not connected so we return nothing
        return nothing
    else
        covered = falses(nnodes)
        covered[1] = true
        route = [1]
        while length(route) < nnodes
            last = route[end]
            infloop = true
            for j in 1 : nnodes
                if !covered[j]
                    if y[last, j] + y[j, last] == 1
                        push!(route, j)
                        covered[j] = true
                        infloop = false
                        break
                    end
                end
            end
            if infloop # infinite loop so we cananot detect a solution here, return nothing
                return nothing
            end
        end
        push!(route, 1) # close the loop
        #build a solution to return
        let
            to = []
            from = []
            cost = 0
            for k in 1 : nnodes
                i, j = route[k], route[k + 1]
                push!(from, i)
                push!(to, j)
                cost += data.cost[i, j]
            end
            return Solution(from, to, cost)
        end
    end
    return nothing
end

function max_flow_min_cut(data::TSPData, x)::Union{Nothing, Set{Int64}}
    nnodes = size(x, 1)
    g = Graphs.SimpleDiGraph(nnodes)
    for i in 1:nnodes, j in 1:nnodes
        if x[i,j] < 1e-5
        x[i,j] = 0
        end
        if x[i,j]> 1e-5
            add_edge!(g,i,j)
        end
    end

    for i in 1:nnodes, j in 1:nnodes
        if i != j
            value = 0
            cut_2=[]
            cut_2 = GraphsFlows.mincut(g,i,j,x,GraphsFlows.PushRelabelAlgorithm())
            for p in cut_2[1], q in cut_2[2]
                value += x[p,q]
            end
            if value <= 1-0.01
                println("max_flow_min_cut found")
                return Set(cut_2[1])
            end
        end
    end
    return nothing
end

using Random
function generate_random(n::Int64, seed::UInt64 = rand(UInt64))::TSPData
    rng = MersenneTwister(seed)
    pos = rand(rng, 0 : 1000, n, 2)
    cost = 1000000 * ones(Int64, n, n)
    for i in 1 : n - 1, j in i + 1 : n
        d = distance(pos[i, 1], pos[i, 2], pos[j, 1], pos[j, 2])
        cost[i, j] = d
        cost[j, i] = d
    end
    return TSPData(pos, cost)
end

data = TravelingSalesman.generate_random(20,UInt(5))

# Solve TSP problem
solution = solve_tsp_dfj(TSPData(data.pos, data.cost))

# Redirect output to file
output_file = "tsp_output.txt"
open(output_file, "w") do io
    redirect_stdout(io) do
        # Print optimal tour sequence and its cost
        solution = solve_tsp_dfj(TSPData(data.pos, data.cost))
        println("From: ", solution.from)
        println("To: ", solution.to)
        println("Cost: ", solution.cost)
    end
end

