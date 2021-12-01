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

# plot results of `D` against `C` for the 1D ring
d_data = readdlm("D-results-matrix.csv", ',')
d_data[1,1] = 4
d_data = Array{Integer}(d_data)
trendline = mean(d_data, dims=2)
c_range = 0.0:0.1:0.7
scatter(c_range, d_data, legend=false, c=:grey, xlabel="C", ylabel="D", markerstrokecolor = :grey,
	xtickfontsize=15, ytickfontsize=15, linewidth=2, xguidefontsize=18, yguidefontsize=18, fmt=:pdf)
plot!(c_range, trendline, c=:black, linewidth=2)
savefig(string("D-C-curve.pdf"))

# plot `E` FI curve
phi_x_train = [-0.015,      0, 0.025, 0.05, 0.075,  0.1, 0.15]
phi_y_train = [     0,  0.005, 0.033, 0.05, 0.06, 0.068, 0.08]
phi_spline = CubicSpline(phi_x_train, phi_y_train)
phi_curve = phi_spline[range(phi_x_train[1], stop=phi_x_train[end], length=1_000)]
phi_curve[phi_curve.<0.0] .= 0.0
xs = range(phi_x_train[1], stop=phi_x_train[end], length=1_000)
plot(xs, phi_curve, legend=false, c=:black, xlabel="I", ylabel="V",
	xtickfontsize=15, ytickfontsize=15, linewidth=2,
	xguidefontsize=18, yguidefontsize=18, fmt=:pdf)
savefig(string("E_F-I-curve.pdf"))

# plot `I` FI curve
function psi(x)
	return max(0, (x-0.05)*0.1)
end
xs_psi = range(0, stop=0.6, length=1_000)
psi_curve = psi.(xs_psi)

plot(xs_psi, psi_curve, legend=false, c=:black, xlabel="I", ylabel="V",
	xtickfontsize=15, ytickfontsize=15, linewidth=2,
	xguidefontsize=18, yguidefontsize=18, fmt=:pdf)
savefig(string("I_F-I-curve.pdf"))

# distance plots in arbitrary graphs

function cd_scatter(seed, g)
	correlations = readdlm(string("seed",seed,"correlations_array_selective.csv"), ',', Float64)
	diam_g = diameter(g)
	geodesics_array = zeros(Int, nv(g), nv(g))
	distances = zeros(nv(g)^2 - nv(g), 2)

	for v in vertices(g)
		geodesics_array[v,:] = gdistances(g,v)
	end

	k = 1
	for i in vertices(g)
		for j in vertices(g)
			if i!=j
				distance = geodesics_array[i,j]
				correlation = correlations[i,j]
				distances[k,1] = distance
				distances[k,2] = correlation
				k+=1
			end
		end
	end

	scatter(distances[:,1], distances[:,2], legend=false, c=:black, xlabel="Distance", ylabel="Correlation")
	savefig(string("seed",seed,"_distances-correlations-scatter.pdf"))
end

# geometric indicies calculations

function geometric_indicies(seed, g, selective)
	if selective
		correlations_array = readdlm(string("seed",seed,"correlations_array_selective.csv"), ',', Float64)
	else
		correlations_array = readdlm(string("seed",seed,"correlations_array.csv"), ',', Float64)
	end
	diam_g = diameter(g)
	graph_size = nv(g)
	geodesics_array = zeros(Int, graph_size, graph_size)
	distances = zeros(nv(g)^2 - nv(g), 2)

	for v in vertices(g)
		geodesics_array[v,:] = gdistances(g,v)
	end

	Rs = zeros(diam_g)

	for d=1:diam_g
		R = 0
		for mu in 1:graph_size
			for v in 1:graph_size
				if (mu != v)
					la = geodesics_array[mu,v]<=d ? 1 : -1
					R += la*correlations_array[mu, v]
				end
			end
		end
		R /= ((graph_size*graph_size)-graph_size)
		Rs[d] = R
	end

	if selective
		open(string("seed",seed,"_Rs_selective.txt"),"a") do io
		   println(io,"Rs ",Rs)
		end
	else
		open(string("seed",seed,"_Rs.txt"),"a") do io
		   println(io,"Rs ",Rs)
		end
	end
end

# examples
seed = 1636988619
g = generate_triK5_graph()
cd_scatter(seed, g)

seed = 1636950525
g = generate_triK5_graph()
geometric_indicies(seed, g, true) # selective neurons
geometric_indicies(seed, g, false) # all neurons
