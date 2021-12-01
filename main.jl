using Logging
using Random
using LinearAlgebra
using StatsBase
using Distributions
using CubicSplines
using LightGraphs
using Plots
using DelimitedFiles
using GraphPlot
using Compose
using Cairo
using HypothesisTests
using Clustering

include("./logger.jl")
ConsoleLogger() |> logger.ElapsedTimeLoggerDecorator |> global_logger
findnearest(A::AbstractArray,t) = findmin(abs.(A.-t))[2] # find the nearest absolute value `t` in `A`

# general parameters

stim_start = 1 # ms
stim_end_time = 80 # ms
inhib_stim_end_time = 80 # ms
tau_exc = 10 # ms (E current decay time constant)
tau_inh_global = 2 # ms (global I current decay time constant)
tau_inh_local = 2 # ms (local I current decay time constant)
E_to_GI_conn_prob = 0.1 # connection probability of E neurons to randomly synapse to global I neurons
GI_to_E_conn_prob = 0.5 # connection probability of global I neurons to randomly synapse to E neurons

phi_x_train = [-0.015,      0, 0.025, 0.05, 0.075,  0.1, 0.15]
phi_y_train = [     0,  0.005, 0.033, 0.05, 0.06, 0.068, 0.08]
phi_spline = CubicSpline(phi_x_train, phi_y_train)
phi_curve = phi_spline[range(phi_x_train[1], stop=phi_x_train[end], length=1_000)]
phi_curve[phi_curve.<0.0] .= 0.0
xs = range(phi_x_train[1], stop=phi_x_train[end], length=1_000)

# graph generator functions

function generate_triK5_graph()
	triK5_adjacency_matrix = readdlm("triK5_adjacency_matrix.csv", ',', Int)
	g = SimpleGraph(triK5_adjacency_matrix)
	return g
end

function generate_karate_club_graph()
	karate_edges = readdlm("karate_edges.csv", ',', Int)
	g = SimpleGraph(maximum(karate_edges))
	for i in 1:size(karate_edges)[1]
		add_edge!(g, karate_edges[i,1], karate_edges[i,2])
	end
	return g
end

function generate_tutte_graph()
	return smallgraph(:tutte)
end

function generate_gridworld_graph()
	g_grid = LightGraphs.grid([5,5])
	h_grid = LightGraphs.grid([5,5])
	gh_grid = blockdiag(g_grid, h_grid)
	i_grid = LightGraphs.grid([5,5])
	j_grid = LightGraphs.grid([5,5])
	ij_grid = blockdiag(i_grid, j_grid)
	ghij_grid = blockdiag(gh_grid, ij_grid)
	add_edge!(ghij_grid, 23, 28)
	add_edge!(ghij_grid, 15, 61)
	add_edge!(ghij_grid, 73, 78)
	add_edge!(ghij_grid, 86, 40)
	return ghij_grid
end

# weight connection functions

function create_XX_graphical_grinasty_connection_matrix(NE, E_patterns, g)
	E_assembly_length = length(findall(isequal(1), E_patterns))
	n_assemblies = Int(length(findall(>(0), E_patterns))/E_assembly_length)
	@assert length(vertices(g)) == n_assemblies
	J = zeros(NE,NE)
	# auto-association
	for a in 1:n_assemblies
		J[findall(isequal(a), E_patterns),findall(isequal(a), E_patterns)] .+= 1
	end
	# hetero-association
	for v in vertices(g)
		for n in neighbors(g, v)
			J[findall(isequal(v), E_patterns),findall(isequal(n), E_patterns)] .+= 1
			J[findall(isequal(n), E_patterns),findall(isequal(v), E_patterns)] .+= 1
		end
	end
	J .-= Diagonal(J) # no self-edges/autopases
	return J
end

function create_XX_graphical_amit_connection_matrix(NE, E_patterns, g, a)
	E_assembly_length = length(findall(isequal(1), E_patterns))
	n_assemblies = Int(length(findall(>(0), E_patterns))/E_assembly_length)
	@assert length(vertices(g)) == n_assemblies
	J = zeros(NE,NE)
	# auto-association
	for a in 1:n_assemblies
		J[findall(isequal(a), E_patterns),findall(isequal(a), E_patterns)] .= 1
	end
	# hetero-association
	for v in vertices(g)
		for n in neighbors(g, v)
			J[findall(isequal(v), E_patterns),findall(isequal(n), E_patterns)] .= a
			J[findall(isequal(n), E_patterns),findall(isequal(v), E_patterns)] .= a
		end
	end
	J .-= Diagonal(J) # no self-edges/autopases
	return J
end

function create_XX_amit_connection_matrix(patterns, a)
	patterns *= 1.0
	pattern_step_idx = circshift(patterns, 1)
	J_same = transpose(patterns)*patterns
	J_same[J_same.>0] .= 1.0
	J_ahead = transpose(pattern_step_idx)*patterns
	J_ahead[J_ahead.>0] .= a
	J_behind = transpose(patterns)*pattern_step_idx
	J_behind[J_behind.>0] .= a
	J = max.(J_same, max.(J_ahead, J_behind))
    J .-= Diagonal(J) # no self-edges/autopases
	return J
end

function create_XX_grinasty_connection_matrix(patterns)
	patterns *= 1.0
	pattern_step_idx = circshift(patterns, 1)
	J_same = transpose(patterns)*patterns
	J_ahead = transpose(pattern_step_idx)*patterns
	J_behind = transpose(patterns)*pattern_step_idx
	J = J_same + J_ahead + J_behind
    J .-= Diagonal(J) # no self-edges/autopases
	return J
end

function create_graphical_local_EI_IE_matrices(E_patterns, I_assemblies, NE, NLI)
	E_pattern_length = length(findall(isequal(1), E_patterns))
	I_assembly_length = length(findall(isequal(1), I_assemblies))
	n_patterns = Int(length(E_patterns)/E_pattern_length)

	E_patterns_idxs = zeros(Int64, E_pattern_length, length(E_patterns))
	I_assemblies_idxs = zeros(Int64, I_assembly_length, length(E_patterns))

	for pattern = 1:n_patterns
		E_patterns_idxs[:,pattern] = findall(isequal(pattern), E_patterns)
		I_assemblies_idxs[:,pattern] = findall(isequal(pattern), I_assemblies)
	end

	J_EI = zeros(NE, NLI)

	for pattern = 1:n_patterns
		J_EI[E_patterns_idxs[:,pattern],I_assemblies_idxs[:,pattern]] = ones(E_pattern_length, I_assembly_length)
	end

	J_IE = deepcopy(J_EI)'

	return J_EI, J_IE
end

function create_local_EI_IE_matrices(E_patterns, I_assemblies)
	E_pattern_length = length(findall(isone.(E_patterns[1,:])))
	I_assembly_length = length(findall(isone.(I_assemblies[1,:])))
	n_patterns = size(E_patterns)[1]

	E_patterns_idxs = zeros(Int64, E_pattern_length,size(E_patterns)[1])
	I_assemblies_idxs = zeros(Int64, I_assembly_length,size(E_patterns)[1])

	for pattern = 1:n_patterns
		E_patterns_idxs[:,pattern] = findall(isone.(E_patterns[pattern,:]))
		I_assemblies_idxs[:,pattern] = findall(isone.(I_assemblies[pattern,:]))
	end

	J_EI = zeros(size(E_patterns)[2], size(I_assemblies)[2])

	for pattern = 2:n_patterns-1
		J_EI[E_patterns_idxs[:,pattern],I_assemblies_idxs[:,pattern]] = ones(E_pattern_length, I_assembly_length)
	end

	pattern = 1
	J_EI[E_patterns_idxs[:,pattern],I_assemblies_idxs[:,pattern]] = ones(E_pattern_length, I_assembly_length)
	pattern = n_patterns
	J_EI[E_patterns_idxs[:,pattern],I_assemblies_idxs[:,pattern]] = ones(E_pattern_length, I_assembly_length)

	J_IE = deepcopy(J_EI)'

	return J_EI, J_IE
end

# neural response functions

function phi(x)
	if x < -0.05
		return 0
	elseif x > 0.15
		return 0.08
	else
		return phi_curve[findnearest(xs, x)]
	end
end

function psi(x)
	return max(0, (x-0.05)*0.1)
end

# main simulation function

function simulate(graphical, graph_name, grinasty, gamma, c, alpha_var, J_EE, J_ELI, J_LIE, J_GIE, J_EGI, E_patterns, I_assemblies, NE, NLI, NGI, sim_length, stim_target, f, target_dishib=false)
	E_current = zeros(NE)
	E_rate = zeros(NE)
	LI_current = zeros(NLI)
	LI_rate = zeros(NLI)
	GI_current = zeros(NGI)
	GI_rate = zeros(NGI)

	E_rates = zeros(Int(sim_length/0.1), NE)
	LI_rates = zeros(Int(sim_length/0.1), NLI)
	GI_rates = zeros(Int(sim_length/0.1), NGI)

	# These factors are to give an 'average' response population-to-population
	# In Amit et al. 1994:
	# - E to E factor given is in eq. 1: 1/(f*N), where f*N=40
	# - E to GI factor given is in eq. 5: 1/N_inh, where N_inh=f*N=40
	# Notice: in Amit et al. 1994: N_exc/p=40
	# These factors were hand-tuned and the tuning procedure is not described.
	# However, the given factors seem to rely on a constant size of E assemblies (40 neurons).
	# currently, these factors rely on p=100
	# In the grinasty case, we multiply f by 1.5 due to Amit rule discounting (1*40)+(0.5*40)+(0.5*40)=80 c.f. Grinasty rule (1*40)+(1*40)+(1*40)=120
	# this multiplication seems to make Amit and Grinasty rules more comparable with the same values of c
	# the fact this multiplication works suggests the original factors are based on the maximum possible sum of indegree weights
	if graphical
		if graph_name == "triK5"
			if grinasty
				E_to_E_factor = 1/(NE*f*2.5) # triK5 has degree 4, so max input is 40*4 (Grinasty heteroassociation) + 40 (autoassociation) = 200, and the Amit rule max is 120, so 200/80=2.5
			else
				E_to_E_factor = 1/(NE*f*1.5) # triK5 has degree 4, so max input is 20*4 (Amit heteroassociation) + 40 (autoassociation) = 120
			end
		elseif graph_name == "karate"
			if grinasty
				E_to_E_factor = 1/(NE*f*2.794) # karate club has avg degree 4.58824, so max input is 40*4.58824 (Grinasty heteroassociation) + 40 (autoassociation) = 223.5296, and original Amit model max is 80, so 223.5296/80=2.794
			else
				E_to_E_factor = 1/(NE*f*1.647) # karate club has avg degree 4.58824, so max input is 20*4.58824 (Amit heteroassociation) + 40 (autoassociation) = 131.764798, 131.764798/80=1.647
			end
		elseif graph_name == "tutte"
			if grinasty
				E_to_E_factor = 1/(NE*f*2) # the tutte graph has degree 3, so max input is 40*3 (Grinasty heteroassociation) + 40 (autoassociation) = 160, and original Amit model max is 80, so 160/80=2
			else
				E_to_E_factor = 1/(NE*f*1.25) # the tutte graph has degree, so max input is 20*3 (Amit heteroassociation) + 40 (autoassociation) = 100, 100/80=1.25
			end
		elseif graph_name == "gridworld"
			if grinasty
				E_to_E_factor = 1/(NE*f*2.14) # the gridworld graph has avg degree 3.28, so max input is 40*3.28 (Grinasty heteroassociation) + 40 (autoassociation) = 171.2, and original Amit model max is 80, so 171.2/80=2.14
			else
				E_to_E_factor = 1/(NE*f*1.32) # the gridworld graph has degree, so max input is 20*3.28 (Amit heteroassociation) + 40 (autoassociation) = 105.6, 105.6/80=1.32
			end
		end
	else
		if grinasty
			E_to_E_factor = 1/(NE*f*1.5)
		else
			E_to_E_factor = 1/(NE*f)
		end
	end
	E_to_GI_factor = 1/(NE*f*E_to_GI_conn_prob)
	E_to_LI_factor = 1/(NLI*f)
	GI_to_E_factor = 1/(NGI*GI_to_E_conn_prob)
	LI_to_E_factor = 1/(NLI*f)

	# we use `gamma` to scale the overall level of inhibition (for testing only -- in the paper, gamma=1)
	# we use `c` to balance between global (c=0) and local (c=1) inhibition
	# we use `alpha_var` to further rescale global inhibition independently of local inhibition (for testing only -- in the paper, alpha=1)
	local_IE_weighting = gamma*c*LI_to_E_factor
	global_IE_weighting = gamma*(1-c*alpha_var)*GI_to_E_factor

	for t = 1:Int(sim_length/0.1)
		# uncomment for progress printing
		# if( (floor(Int, t*100.0))%10000==0 )
		# 	@info "Simulating timestep" t
		# end
		LI_H = zeros(NLI)
		if Int(stim_start/0.1) <= t <= Int(stim_end_time/0.1)
			if graphical
				E_H = zeros(NE)
				E_H[findall(isequal(stim_target), E_patterns)] .= 0.2
			else
				E_H = E_patterns[stim_target,:].*(0.2)
			end
		else
			E_H = zeros(NE)
		end

		# neural dyanmics
		E_current += (-E_current + E_to_E_factor*(J_EE*E_rate) .- local_IE_weighting*(J_LIE'*LI_rate) .- global_IE_weighting*(J_GIE'*GI_rate) + E_H)/tau_exc
		E_rate = phi.(E_current) .+ abs.(rand(Normal(0.0, 0.00015), NE))
		LI_current += (-LI_current .+ E_to_LI_factor*(J_ELI'*E_rate) + LI_H)/tau_inh_local
		LI_rate = psi.(LI_current)
		GI_current += (-GI_current .+ E_to_GI_factor*(J_EGI'*E_rate))/tau_inh_global
		GI_rate = psi.(GI_current)

		E_rates[t, :] = E_rate
		LI_rates[t, :] = LI_rate
		GI_rates[t, :] = GI_rate
	end

	return E_rates, LI_rates, GI_rates
end

# main function for a single simulation

function main(gamma, c, alpha_var, rebalanced, grinasty, graphical, graph_name, sim_length=250, p=100::Int, NE=4_000::Int, NLI=500::Int, NGI=500::Int, a=0.5::Float64, f=0.01::Float64, target_dishib=false)
	seed = round(Int, time())
	@info "Random" seed
	Random.seed!(seed)

	@info "Creating patterns"
	if graphical
		if graph_name == "triK5"
			g = generate_triK5_graph()
		elseif graph_name == "karate"
			g = generate_karate_club_graph()
		elseif graph_name == "tutte"
			g = generate_tutte_graph()
		elseif graph_name == "gridworld"
			g = generate_gridworld_graph()
		end

		p = length(vertices(g))
		E_patterns_length = Int(f*NE)
		E_patterns = vcat(fill.(collect(1:p),E_patterns_length)...)

		I_assembly_length = Int(f*NLI)
		I_assemblies = vcat(fill.(collect(1:p),I_assembly_length)...)

		@info "Creating connection matrices"
		if grinasty
			J_EE = create_XX_graphical_grinasty_connection_matrix(NE, E_patterns, g)
		else
			J_EE = create_XX_graphical_amit_connection_matrix(NE, E_patterns, g, a)
		end
		J_ELI, J_LIE = create_graphical_local_EI_IE_matrices(E_patterns, I_assemblies, NE, NLI)
	else
		E_patterns = zeros(p, NE)
		E_pattern_length = Int(f*NE)
		for E_pattern = 1:p
			E_patterns[E_pattern, randperm(NE)[1:E_pattern_length]] .= 1
		end
		I_assemblies = zeros(p, NLI)
		I_assembly_length = Int(f*NLI)
		for I_assembly = 1:p
			I_assemblies[I_assembly, randperm(NLI)[1:I_assembly_length]] .= 1
		end

		@info "Creating connection matrices"
		if grinasty
			J_EE = create_XX_grinasty_connection_matrix(E_patterns)
		else
			J_EE = create_XX_amit_connection_matrix(E_patterns, a)
		end
		J_ELI, J_LIE = create_local_EI_IE_matrices(E_patterns, I_assemblies)
		g = nothing
	end

	J_GIE = rand(Categorical([(1-GI_to_E_conn_prob), GI_to_E_conn_prob]), NGI, NE) .- 1
	J_EGI = rand(Categorical([(1-E_to_GI_conn_prob), E_to_GI_conn_prob]), NE, NGI) .- 1

	if rebalanced
		# reweight the global and local I such that it is proportionally more on each E neuron which receives proprtionally more E
		sum_of_inputs = sum(J_EE, dims=1)
		mean_nonzero = sum(J_EE)/length(sum_of_inputs[sum_of_inputs.>0])
		weighted_sum_of_inputs = sum_of_inputs/mean_nonzero
		weighted_sum_of_inputs = dropdims(weighted_sum_of_inputs, dims=1)
		J_LIE = weighted_sum_of_inputs'.*J_LIE
		J_GIE = weighted_sum_of_inputs'.*J_GIE
	end

	@info "Starting simulation"
	E_rates, LI_rates, GI_rates = simulate(graphical, graph_name, grinasty, gamma, c, alpha_var, J_EE, J_ELI, J_LIE, J_GIE, J_EGI, E_patterns, I_assemblies, NE, NLI, NGI, sim_length, 1, f, target_dishib)
	@info "Simulation complete"
	return g, E_rates, LI_rates, GI_rates, J_EE, J_ELI, J_LIE, E_patterns, I_assemblies, seed
end

# plotting functions

function make_simple_figure()
	stim_neurons = findall(>(0), E_patterns[1,:])
	neighbour1 = findall(>(0), E_patterns[2,:])
	neighbour2 = findall(>(0), E_patterns[3,:])
	neighbour3 = findall(>(0), E_patterns[4,:])
	other_neurons = findall(<(1), E_patterns[1,:])

	stimed_rates = E_rates[:,stim_neurons]
	neighbour1_rates = E_rates[:,neighbour1]
	neighbour2_rates = E_rates[:,neighbour2]
	neighbour3_rates = E_rates[:,neighbour3]
	other_rates = E_rates[:,other_neurons]

	mean_stim_rates = mean(stimed_rates, dims=2)
	std_stim_rates = std(stimed_rates, dims=2)

	mean_neighbour1_rates = mean(neighbour1_rates, dims=2)
	std_neighbour1_rates = std(neighbour1_rates, dims=2)

	mean_neighbour2_rates = mean(neighbour2_rates, dims=2)
	std_neighbour2_rates = std(neighbour2_rates, dims=2)

	mean_neighbour3_rates = mean(neighbour3_rates, dims=2)
	std_neighbour3_rates = std(neighbour3_rates, dims=2)

	mean_other_rates = mean(other_rates, dims=2)
	std_other_rates = mean(other_rates, dims=2)

	plot([mean_stim_rates mean_stim_rates], grid=true, ylims=[0, 0.09], label=["Stimulated pattern" ""],
	    fillrange=[mean_stim_rates+std_stim_rates mean_stim_rates-std_stim_rates], fillalpha=0.3, c=:green)
	plot!([mean_neighbour1_rates mean_neighbour1_rates], grid=true, ylims=[0, 0.09], label=["1st neighbour" ""],
		fillrange=[mean_neighbour1_rates+std_neighbour1_rates mean_neighbour1_rates-std_neighbour1_rates], fillalpha=0.3, c=:orange4)
	plot!([mean_neighbour2_rates mean_neighbour2_rates], grid=true, ylims=[0, 0.09], label=["2nd neighbour" ""],
		fillrange=[mean_neighbour2_rates+std_neighbour2_rates mean_neighbour2_rates-std_neighbour2_rates], fillalpha=0.3, c=:orange3)
	plot!([mean_neighbour3_rates mean_neighbour3_rates], grid=true, ylims=[0, 0.09], label=["3rd neighbour" ""],
		fillrange=[mean_neighbour3_rates+std_neighbour3_rates mean_neighbour3_rates-std_neighbour3_rates], fillalpha=0.3, c=:orange2)
	plot!([mean_other_rates mean_other_rates], grid=true, ylims=[0, 0.09], label=["Other Patterns" ""],
	    fillrange=[mean_other_rates+std_other_rates mean_other_rates-std_other_rates], fillalpha=0.3, c=:orange,
	    xlabel="Time (ms)", ylabel=string("Spiking rate (mean ± S.D.)"), legend=:bottomright, fmt=:pdf)
end

function make_simple_inhib_figure()
	stim_neurons = findall(isequal(1), I_assemblies)
	neighbour1 = findall(isequal(2), I_assemblies)
	neighbour2 = findall(isequal(3), I_assemblies)
	neighbour3 = findall(isequal(4), I_assemblies)
	other_neurons = findall(>(4), I_assemblies)

	stimed_rates = LI_rates[:,stim_neurons]
	neighbour1_rates = LI_rates[:,neighbour1]
	neighbour2_rates = LI_rates[:,neighbour2]
	neighbour3_rates = LI_rates[:,neighbour3]
	other_rates = LI_rates[:,other_neurons]

	mean_stim_rates = mean(stimed_rates, dims=2)
	std_stim_rates = std(stimed_rates, dims=2)

	mean_neighbour1_rates = mean(neighbour1_rates, dims=2)
	std_neighbour1_rates = std(neighbour1_rates, dims=2)

	mean_neighbour2_rates = mean(neighbour2_rates, dims=2)
	std_neighbour2_rates = std(neighbour2_rates, dims=2)

	mean_neighbour3_rates = mean(neighbour3_rates, dims=2)
	std_neighbour3_rates = std(neighbour3_rates, dims=2)

	mean_other_rates = mean(other_rates, dims=2)
	std_other_rates = mean(other_rates, dims=2)

	plot([mean_stim_rates mean_stim_rates], grid=true, ylims=[0, 0.09], label=["Stimulated pattern" ""],
	    fillrange=[mean_stim_rates+std_stim_rates mean_stim_rates-std_stim_rates], fillalpha=0.3, c=:green)
	plot!([mean_neighbour1_rates mean_neighbour1_rates], grid=true, ylims=[0, 0.09], label=["1st neighbour" ""],
		fillrange=[mean_neighbour1_rates+std_neighbour1_rates mean_neighbour1_rates-std_neighbour1_rates], fillalpha=0.3, c=:orange4)
	plot!([mean_neighbour2_rates mean_neighbour2_rates], grid=true, ylims=[0, 0.09], label=["2nd neighbour" ""],
		fillrange=[mean_neighbour2_rates+std_neighbour2_rates mean_neighbour2_rates-std_neighbour2_rates], fillalpha=0.3, c=:orange3)
	plot!([mean_neighbour3_rates mean_neighbour3_rates], grid=true, ylims=[0, 0.09], label=["3rd neighbour" ""],
		fillrange=[mean_neighbour3_rates+std_neighbour3_rates mean_neighbour3_rates-std_neighbour3_rates], fillalpha=0.3, c=:orange2)
	plot!([mean_other_rates mean_other_rates], grid=true, ylims=[0, 0.09], label=["Other Patterns" ""],
	    fillrange=[mean_other_rates+std_other_rates mean_other_rates-std_other_rates], fillalpha=0.3, c=:orange,
	    xlabel="Time (ms)", ylabel=string("Spiking rate (mean ± S.D.)"), legend=:bottomright, fmt=:pdf)
end

function make_graph_figure(save_string, savefig_bool, graph_name)
	n_vertices = length(vertices(g))
	vertex_activities = zeros(n_vertices)
	for v in vertices(g)
		vertex_activities[v] = mean(E_rates[end-20:end,(1+(v-1)*40):(v*40)])
	end

	nodelabel = 1:nv(g)
	membership = vertex_activities./0.09
	nodecolor = cgrad(:grays, rev=true)
	# membership color
	nodefillc = nodecolor[membership.*256]

	if (graph_name == "triK5")
		locs_x = zeros(length(collect(vertices(g))))
		locs_y = zeros(length(collect(vertices(g))))
		locs_y[1:5] .+= 0.5
		locs_y[6:15] .-= 0.5
		locs_x[6:10] .-= 0.5
		locs_x[11:15] .+= 0.5

		# community A
		locs_x[2] += 0.2
		locs_x[3] -= 0.2
		locs_x[4] += 0.15
		locs_x[5] -= 0.15
		locs_y[1] += 0.8
		locs_y[2:3] .+= 0.5
		locs_y[4:5] .+= 0.1

		# community B
		locs_x[9] += 0.2
		locs_x[8] -= 0.2
		locs_x[10] += 0.15
		locs_x[7] -= 0.15
		locs_y[6] += 0.8
		locs_y[9] += 0.5
		locs_y[8] += 0.5
		locs_y[10] += 0.1
		locs_y[7] += 0.1

		# community C
		locs_x[12] += 0.2
		locs_x[15] -= 0.2
		locs_x[13] += 0.15
		locs_x[14] -= 0.15
		locs_y[11] += 0.8
		locs_y[12] += 0.5
		locs_y[15] += 0.5
		locs_y[13] += 0.1
		locs_y[14] += 0.1

		draw(PDF(save_string), gplot(g, locs_x, locs_y, nodefillc=nodefillc))
	elseif (graph_name == "karate")
		karate_plot_xy = readdlm("karate_plot_xy.txt", '\t', Float64)
		karate_plot_x = karate_plot_xy[1,:]
		karate_plot_y = karate_plot_xy[2,:]
		draw(PDF(save_string), gplot(g, karate_plot_x, karate_plot_y, nodefillc=nodefillc))
	elseif (graph_name == "tutte")
		tutte_plot_xy = readdlm("tutte_plot_xy.txt", '\t', Float64)
		tutte_plot_x = tutte_plot_xy[1,:]
		tutte_plot_y = tutte_plot_xy[2,:]
		draw(PDF(save_string), gplot(g, tutte_plot_x, tutte_plot_y, nodefillc=nodefillc))
	elseif (graph_name == "gridworld")
		locs_x = zeros(100)
		locs_y = zeros(100)
		for v in 1:25
			locs_x[v] = mod(v-1, 5)/4
			locs_y[v] = floor((v/5)+1)/4
			if (v % 5) == 0
				locs_y[v] -= 0.25
			end
		end
		locs_x[1:25] .-= 1
		locs_y[1:25] .-= 1

		for v in 26:50
			locs_x[v] = mod(v-1, 5)/4
			locs_y[v] = floor((v/5)+1)/4
			if (v % 5) == 0
				locs_y[v] -= 0.25
			end
		end
		locs_x[26:50] .-= 1
		locs_y[26:50] .-= 0.5

		for v in 51:75
			locs_x[v] = mod(v-1, 5)/4
			locs_y[v] = floor((v/5)+1)/4
			if (v % 5) == 0
				locs_y[v] -= 0.25
			end
		end
		locs_x[51:75] .+= 0.5
		locs_y[51:75] .-= 3.5

		for v in 76:100
			locs_x[v] = mod(v-1, 5)/4
			locs_y[v] = floor((v/5)+1)/4
			if (v % 5) == 0
				locs_y[v] -= 0.25
			end
		end
		locs_x[76:100] .+= 0.5
		locs_y[76:100] .-= 3

		draw(PDF(save_string), gplot(g, locs_x, locs_y, nodefillc=nodefillc))
	else
		if savefig_bool
			draw(PDF(save_string), gplot(g, nodefillc=nodefillc, nodelabel=nodelabel))
		else
			gplot(g, layout=circular_layout, nodefillc=nodefillc, nodelabel=nodelabel)
		end
	end
end

# example simulations and plots

simlen = 500
c = 0.7
f = 0.02
p = 50
g, E_rates, LI_rates, GI_rates, J_EE, J_ELI, J_LIE, E_patterns, I_assemblies, seed = main(1, c, 1.0, true, true, false, "", simlen, p, 4_000, 500, 500, 0, f, false)
make_simple_figure()

savefig_bool = true
simlen = 500
graph_name = "gridworld"
c = 0.525
g, E_rates, LI_rates, GI_rates, J_EE, J_ELI, J_LIE, E_patterns, I_assemblies, seed = main(1, c, 1.0, true, true, true, graph_name, simlen, 100, 4_000, 500, 500, 0, 0.01, false)
save_string = string("seed",seed,"_c",c,"_graph",graph_name,"_simlen",simlen,"_grinasty_normd E rates on vertices.pdf")
plot(GI_rates,legend=false, xlabel="Time (ms)", ylabel=string("Spiking rate"), color=:grey, ylims=[0,0.35])
savefig(string("seed",seed,"_global I rates.pdf"))
plot(LI_rates,legend=false, xlabel="Time (ms)", ylabel=string("Spiking rate"), color=:grey, ylims=[0,0.065])
savefig(string("seed",seed,"_local I rates.pdf"))
make_graph_figure(save_string, savefig_bool, graph_name)

# main correlations simulations function

function main_correlations(gamma, c, alpha_var, rebalanced, grinasty, graphical, graph_name, sim_length=250, p=100::Int, NE=4_000::Int, NLI=500::Int, NGI=500::Int, a=0.5::Float64, f=0.01::Float64, target_dishib=false)
	seed = round(Int, time())
	@info "Random" seed
	Random.seed!(seed)

	@info "Creating patterns"
	if graphical
		if graph_name == "triK5"
			g = generate_triK5_graph()
		elseif graph_name == "karate"
			g = generate_karate_club_graph()
		elseif graph_name == "tutte"
			g = generate_tutte_graph()
		elseif graph_name == "gridworld"
			g = generate_gridworld_graph()
		end

		p = length(vertices(g))
		E_patterns_length = Int(f*NE)
		E_patterns = vcat(fill.(collect(1:p),E_patterns_length)...)

		I_assembly_length = Int(f*NLI)
		I_assemblies = vcat(fill.(collect(1:p),I_assembly_length)...)

		@info "Creating connection matrices"
		if grinasty
			J_EE = create_XX_graphical_grinasty_connection_matrix(NE, E_patterns, g)
		else
			J_EE = create_XX_graphical_amit_connection_matrix(NE, E_patterns, g, a)
		end
		J_ELI, J_LIE = create_graphical_local_EI_IE_matrices(E_patterns, I_assemblies, NE, NLI)
	else
		E_patterns = zeros(p, NE)
		E_pattern_length = Int(f*NE)
		for E_pattern = 1:p
			E_patterns[E_pattern, randperm(NE)[1:E_pattern_length]] .= 1
		end
		I_assemblies = zeros(p, NLI)
		I_assembly_length = Int(f*NLI)
		for I_assembly = 1:p
			I_assemblies[I_assembly, randperm(NLI)[1:I_assembly_length]] .= 1
		end

		@info "Creating connection matrices"
		if grinasty
			J_EE = create_XX_grinasty_connection_matrix(E_patterns)
		else
			J_EE = create_XX_amit_connection_matrix(E_patterns, a)
		end
		J_ELI, J_LIE = create_local_EI_IE_matrices(E_patterns, I_assemblies)
		g = nothing
	end

	J_GIE = rand(Categorical([(1-GI_to_E_conn_prob), GI_to_E_conn_prob]), NGI, NE) .- 1
	J_EGI = rand(Categorical([(1-E_to_GI_conn_prob), E_to_GI_conn_prob]), NE, NGI) .- 1

	if rebalanced
		# reweight the global and local I such that it is proportional for each E neuron with its incoming E
		sum_of_inputs = sum(J_EE, dims=1)
		mean_nonzero = sum(J_EE)/length(sum_of_inputs[sum_of_inputs.>0])
		weighted_sum_of_inputs = sum_of_inputs/mean_nonzero
		weighted_sum_of_inputs = dropdims(weighted_sum_of_inputs, dims=1)
		J_LIE = weighted_sum_of_inputs'.*J_LIE
		J_GIE = weighted_sum_of_inputs'.*J_GIE
	end

	@info "Starting simulations"
	max_rep = 1
	rep_attractors = zeros(max_rep, p, NE)
	for rep = 1:max_rep
		@info "Starting simulation repetition" rep
		for stim_id = 1:p
			E_rates, LI_rates, GI_rates = E_rates, LI_rates, GI_rates = simulate(graphical, graph_name, grinasty, gamma, c, alpha_var, J_EE, J_ELI, J_LIE, J_GIE, J_EGI, E_patterns, I_assemblies, NE, NLI, NGI, sim_length, stim_id, f, target_dishib)
			rep_attractors[rep,stim_id,:] = mean(E_rates[end-Int(19/0.1):end,:],dims=1)
		end
	end
	attractors = dropdims(mean(rep_attractors,dims=1),dims=1)

	@info "Removing 'non-selective' cells"
	maximum_FRs_in_attractors_all_neurons = maximum(attractors,dims=1)
	histogram([maximum_FRs_in_attractors_all_neurons...], legend=false, xlabel="Maximum firing rate in attractor state of any attractor", ylabel="Number of neurons", fmt=:pdf)
	if graphical
		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_histogram_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_histogram_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_histogram_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_histogram_unbalanced.pdf"))
			end
		end
	else
		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_histogram_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_histogram_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_histogram_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_maximum_FRs_in_attractors_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_histogram_unbalanced.pdf"))
			end
		end
	end
	sig_units_refs = findall(>(0.02), maximum_FRs_in_attractors_all_neurons)
	sig_units_IDs = LinearIndices(maximum_FRs_in_attractors_all_neurons)[sig_units_refs]
	sig_units_FRs = maximum_FRs_in_attractors_all_neurons[sig_units_IDs]
	sig_units_attractors = attractors[:,sig_units_IDs]

	@info "Calculating correlations"
	# all neurons
	correlations_array = zeros(p,p)
	attractor_array_analysed = attractors
	N_s = length(attractor_array_analysed)/p
	for mu = 1:p
		for ve = 1:p
			V_mu_bar = 1/N_s * sum(attractor_array_analysed[mu,:]) # "mean"
			V_mu_bar_sq = 1/N_s * sum(attractor_array_analysed[mu,:].^2) - V_mu_bar^2 # "variance"
			V_ve_bar = 1/N_s * sum(attractor_array_analysed[ve,:])
			V_ve_bar_sq = 1/N_s * sum(attractor_array_analysed[ve,:].^2) - V_ve_bar^2
			COV_mu_ve = 1/N_s * sum(attractor_array_analysed[mu,:].*attractor_array_analysed[ve,:]) - V_mu_bar*V_ve_bar # "covariance"
			C_mu_ve = COV_mu_ve / sqrt(V_mu_bar_sq*V_ve_bar_sq) # "correlation"
			correlations_array[mu,ve] = C_mu_ve
		end
	end
	writedlm(string("seed",seed,"correlations_array.csv"), correlations_array, ", ")

	if graphical
		heatgray = cgrad(:greys, rev=true)
		if (graph_name == "karate") && (c == 0.1)
			clustering_result = kmeans(correlations_array,5)
			idx = sortperm(assignments(clustering_result))
			members = correlations_array[:,idx]
			members = members[idx,:]
			plot(heatmap(members, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		elseif (graph_name == "karate") && (c == 0.525)
			clustering_result = kmeans(correlations_array,3)
			idx = sortperm(assignments(clustering_result))
			members = correlations_array[:,idx]
			members = members[idx,:]
			plot(heatmap(members, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		elseif (graph_name == "tutte") && (c == 0.1)
			clustering_result = kmeans(correlations_array,3)
			idx = sortperm(assignments(clustering_result))
			members = correlations_array[:,idx]
			members = members[idx,:]
			plot(heatmap(members, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		elseif (graph_name == "tutte") && (c == 0.525)
			clustering_result = kmeans(correlations_array,4)
			idx = sortperm(assignments(clustering_result))
			members = correlations_array[:,idx]
			members = members[idx,:]
			plot(heatmap(members, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		else
			plot(heatmap(correlations_array, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		end

		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_unbalanced.pdf"))
			end
		end

		Q = 0
		g_labels, _ = label_propagation(g)
		for mu in 1:p
			for v in 1:p
				if (mu != v)
					lpa = g_labels[mu]==g_labels[v] ? 1 : -1
					Q += lpa*correlations_array[mu, v]
				end
			end
		end
		Q /= ((p*p)-p)
		open(string("seed",seed,"_Q.txt"),"a") do io
		   println(io,"Q ",Q)
		end

	else
		attractor_correlations = zeros(p,p)
		for attractor = 1:p
			attractor_correlations[attractor,:] = diag(circshift(correlations_array, attractor-1))
		end
		plot(collect(0:Int(p/2)-1),attractor_correlations[1:Int(p/2),:],color="grey",linewidth=1)
		plot!(collect(1:Int(p/2)),reverse(attractor_correlations[Int(p/2)+1:p,:],dims=1),color="grey",linewidth=1)
		plot!(collect(0:Int(p/2)),mean(attractor_correlations[1:Int(p/2)+1,:]',dims=1)',color="black",linewidth=4,legend=false,
				xlabel="Attractor distance",ylabel="Attractor correlation (all neurons)")
		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_all_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_unbalanced.pdf"))
			end
		end
	end
	# 'selective' neurons
	correlations_array = zeros(p,p)
	attractor_array_analysed = sig_units_attractors
	N_s = length(attractor_array_analysed)/p
	for mu = 1:p
		for ve = 1:p
			V_mu_bar = 1/N_s * sum(attractor_array_analysed[mu,:]) # "mean"
			V_mu_bar_sq = 1/N_s * sum(attractor_array_analysed[mu,:].^2) - V_mu_bar^2 # "variance"
			V_ve_bar = 1/N_s * sum(attractor_array_analysed[ve,:])
			V_ve_bar_sq = 1/N_s * sum(attractor_array_analysed[ve,:].^2) - V_ve_bar^2
			COV_mu_ve = 1/N_s * sum(attractor_array_analysed[mu,:].*attractor_array_analysed[ve,:]) - V_mu_bar*V_ve_bar # "covariance"
			C_mu_ve = COV_mu_ve / sqrt(V_mu_bar_sq*V_ve_bar_sq) # "correlation"
			correlations_array[mu,ve] = C_mu_ve
		end
	end
	writedlm(string("seed",seed,"correlations_array_selective.csv"), correlations_array, ", ")

	if graphical
		heatgray = cgrad(:greys, rev=true)
		plot(heatmap(correlations_array, xlabel="Vertex",ylabel="Vertex",color = heatgray, clim=(-1,1)))
		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_graph",graph_name,"_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_unbalanced.pdf"))
			end
		end

		Q_selective = 0
		g_labels, _ = label_propagation(g)
		for mu in 1:p
			for v in 1:p
				if (mu != v)
					lpa = g_labels[mu]==g_labels[v] ? 1 : -1
					Q_selective += lpa*correlations_array[mu, v]
				end
			end
		end
		Q_selective /= ((p*p)-p)
		open(string("seed",seed,"_Q_selective.txt"),"a") do io
		   println(io,"Q_selective ",Q_selective)
		end
	else
		attractor_correlations = zeros(p,p)
		for attractor = 1:p
			attractor_correlations[attractor,:] = diag(circshift(correlations_array, attractor-1))
		end
		plot(collect(0:Int(p/2)-1),attractor_correlations[1:Int(p/2),:],color="grey",linewidth=1)
		plot!(collect(1:Int(p/2)),reverse(attractor_correlations[Int(p/2)+1:p,:],dims=1),color="grey",linewidth=1)
		plot!(collect(0:Int(p/2)),mean(attractor_correlations[1:Int(p/2)+1,:]',dims=1)',color="black",linewidth=4,legend=false,
				xlabel="Attractor distance",ylabel="Attractor correlation (all neurons)")
		if rebalanced
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_rebalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_rebalanced.pdf"))
			end
		else
			if grinasty
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_grinasty_unbalanced.pdf"))
			else
				savefig(string("seed",seed,"_attractor_correlations_over_distance_selective_neurons_c",c,"_gamma",gamma,"_alpha",alpha_var,"_NE",NE,"_NGI",NGI,"_NLI",NLI,"_a",a,"_unbalanced.pdf"))
			end
		end
	end
end

# examples (warning: compute time is linear in the number of vertices of the memory graph)

main_correlations(1, 0.1, 1.0, true, true, true, "triK5", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.525, 1.0, true, true, true, "triK5", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.1, 1.0, true, true, true, "tutte", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.525, 1.0, true, true, true, "tutte", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.1, 1.0, true, true, true, "karate", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.525, 1.0, true, true, true, "karate", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.1, 1.0, true, true, true, "gridworld", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
main_correlations(1, 0.525, 1.0, true, true, true, "gridworld", 500, 0, 4000, 500, 500, 0.0, 0.01, false)
