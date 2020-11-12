### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 39d68b7e-24be-11eb-3f02-ebaf79440c32
begin
	# Packages we will use throughout this notebook
	using UMAP
	using Makie
	using XLSX
	using VegaDatasets
	using DataFrames
	using MultivariateStats
	using RDatasets
	using StatsBase
	using Statistics
	using LinearAlgebra
	using Plots
	using ScikitLearn
	using MLBase
	using Distances
end

# ╔═╡ 222f6650-24be-11eb-02d6-1f7db4b8fd5c
md"""
## Dimensionality Reduction
As the name says, dimensionality reduction is the idea of reducing your feature set to a much smaller number. Dimensionality reduction is often used in visualization of datasets to try and detect samples that are similar. We will cover three dimensionality reduction techniques here: 
1. t-SNE
2. PCA
3. umap
"""

# ╔═╡ 08d8fbc0-24c4-11eb-1ef2-61fbf25e9328
md"We will use a dataset from the VegaDatasets package. The dataset is about car specifications of over 400 car models."

# ╔═╡ 11cc3e40-24c4-11eb-2255-c97e7bc507ad
C = DataFrame(VegaDatasets.dataset("Cars"))

# ╔═╡ 29169d70-24c4-11eb-1671-f122fa8fefc5
dropmissing!(C)

# ╔═╡ 3fe8e670-24c4-11eb-3d1c-71d530dc341e
M = Matrix(C[:, 2: 7])

# ╔═╡ 4966dcbe-24c4-11eb-18e4-35ea410c9f13
names(C)

# ╔═╡ 5eb4cfb0-24c4-11eb-0829-679ddd8141f1
begin
	car_origin = C[!, :Origin]
	carmap = labelmap(car_origin) # from MLBase
	uniqueid = labelencode(carmap, car_origin)
end

# ╔═╡ 9fbff700-24c4-11eb-0e18-4f2cac198d02
md"""
### 1️⃣ PCA 
We will first center the data.
"""

# ╔═╡ e57dd0ee-24c4-11eb-20c2-ffbbfd17924b
md"""
###### Normalization(Z-score)
$x' = \frac{x - \mu}{\sigma}$
"""

# ╔═╡ a7fba94e-24c4-11eb-08bd-fd17f911adfd
begin
	# center and normalize the data
	data = M
	data = (data .- mean(data, dims=1)) ./ std(data, dims=1)
end

# ╔═╡ 18bb2fd0-24c5-11eb-17d1-2136b7554f54
md"✒️ PCA expects each column to be an observation, so we will use the transpose of the matrix."

# ╔═╡ 1deadaa0-24c5-11eb-0a30-fb33052e81f3
# each car is now a column, PCA takes features - by - samples matrix
data'

# ╔═╡ 7165b420-24c5-11eb-202c-55d63b9c9d68
md"First, we will fit the model via PCA. `maxoutdim` is the output dimensions, we want it to be 2 in this case."

# ╔═╡ 78a83960-24c5-11eb-0a79-7fc56fbb20e0
p = fit(PCA, data', maxoutdim=2)

# ╔═╡ c1a6ead2-24e3-11eb-060d-e3998d02cf74
md"We can obtain the projection matrix by calling the function `projection`"

# ╔═╡ c8cac480-24e3-11eb-1cb3-87ad5ad98e90
P = projection(p)

# ╔═╡ ce72857e-24e3-11eb-0821-b190f3f5239a
md"Now that we have the projection matrix, `P`, we can apply it on one car as follows:"

# ╔═╡ d423bc60-24e3-11eb-2874-71be233234ee
P' * (data[1, :] - mean(p))

# ╔═╡ ea287c30-24e3-11eb-0338-c3ca0858b598
md"Or we can transorm all the data via the transform function."

# ╔═╡ ec2efa90-24e3-11eb-1275-2d2d836f93f5
Yte = MultivariateStats.transform(p, data') #notice that Yte[:,1] is the same as P'*(data[1,:]-mean(p))

# ╔═╡ f8987810-24e3-11eb-21e9-1571dec657fd
md"We can also go back from two dimensions to 6 dimensions, via the reconstruct function... But this time, it will be approximate."

# ╔═╡ 0a878890-24e4-11eb-19ff-1994dda9d79a
# reconstruct testing observations (approximately)
Xr = reconstruct(p, Yte)

# ╔═╡ 14ea60a0-24e4-11eb-24fe-51cbe2923f63
norm(Xr - data') # this won't be zero

# ╔═╡ 2a0b9e40-24e4-11eb-267b-6fb950873ceb
md"Finally, we will generate a scatter plot of the cars:"

# ╔═╡ 31bdfc52-24e4-11eb-0868-6d2f5133b5f5
Plots.scatter(Yte[1,:], Yte[2,:])

# ╔═╡ 6de045ce-24e4-11eb-1f46-ef1aa2251558
let
	Plots.scatter(Yte[1, car_origin .== "USA"], Yte[2, car_origin .== "USA"], label="USA")
	Plots.xlabel!("pca component1")
	Plots.ylabel!("pca component2")
	Plots.scatter!(Yte[1, car_origin .== "Japen"], Yte[2, car_origin .== "Japen"], label="Japen")
	Plots.scatter!(Yte[1, car_origin .== "Europe"], Yte[2, car_origin .== "Europe"], label="Europe")
end

# ╔═╡ f7e75b10-24e4-11eb-2229-9b95f98bb39e
md"📘 This is interesting! There seems to be three main clusters with cars from the US dominating two clusters."

# ╔═╡ 27d62b2e-24e5-11eb-27a7-892a41315676
let
	p = fit(PCA, data', maxoutdim=3)
	Yte = MultivariateStats.transform(p, data')
	scatter3d(Yte[1, :], Yte[2, :], Yte[3, :], color=uniqueid, legend=false)
end

# ╔═╡ 266da510-24e6-11eb-2287-1b013023b0a6
md"""
### 2️⃣ t-SNE
The next method we will use for dimensionality reduction is t-SNE. There are multiple ways you can call t-SNE from julia. Check out this [notebook](https://github.com/nassarhuda/JuliaTutorials/blob/master/TSNE/TSNE.ipynb). 

But we will take this opportunity to try out something new... Call a function from the Scikit learn python package. This makes use of the package `ScikitLearn`.
"""

# ╔═╡ 37773790-24e6-11eb-2c64-b7218251937a
@sk_import manifold : TSNE

# ╔═╡ 6f6c98e0-24e9-11eb-0cf2-710690aef25b
let
	tfn = TSNE(n_components=2 ,perplexity=20.0, early_exaggeration=50)
	Y2 = tfn.fit_transform(data)
	Plots.scatter(Y2[:,1], Y2[:,2], color=uniqueid, legend=false, size=(400,300), markersize=3)
end

# ╔═╡ e1da4490-24e9-11eb-14cd-7d6890115ab8
md"This is interesting! The same patterns appears to hold here too. "

# ╔═╡ e352dad0-24e9-11eb-1c1d-1d900aa88432
md"""
### 3️⃣ Next, UMAP
This will be our final dimensionality reduction method and we will use the package `UMAP` for it.
"""

# ╔═╡ ea9a6920-24e9-11eb-1612-9fbb5688b457
let
	L = cor(data, data, dims=2)
	embedding = umap(L, 2)
	Plots.scatter(embedding[1,:], embedding[2,:], color=uniqueid)
end

# ╔═╡ 282ff1b0-24ea-11eb-3386-29c8d6df88fb
md"For UMAP, we can create distances between every pair of observations differently, if we choose to. But even with both choices, we will see that UMAP generates a very similar pattern to what we have observed with t-SNE and PCA."

# ╔═╡ 2974f4d0-24ea-11eb-0b28-610bd4a077d4
let
	L = pairwise(Euclidean(), data, data, dims=1) 
	embedding = umap(-L, 2)
	Plots.scatter(embedding[1,:], embedding[2,:], color=uniqueid)
end

# ╔═╡ 3b1d5f60-24ea-11eb-1e8c-b14da03c54ef
md"""
# Finally...
After finishing this notebook, you should be able to:
- [ ] apply tsne on your data
- [ ] apply umap on your data
- [ ] apply pca on your data
- [ ] generate a 3d plot
- [ ] call a function from Python's ScikitLearn
"""

# ╔═╡ Cell order:
# ╟─222f6650-24be-11eb-02d6-1f7db4b8fd5c
# ╠═39d68b7e-24be-11eb-3f02-ebaf79440c32
# ╟─08d8fbc0-24c4-11eb-1ef2-61fbf25e9328
# ╠═11cc3e40-24c4-11eb-2255-c97e7bc507ad
# ╠═29169d70-24c4-11eb-1671-f122fa8fefc5
# ╠═3fe8e670-24c4-11eb-3d1c-71d530dc341e
# ╠═4966dcbe-24c4-11eb-18e4-35ea410c9f13
# ╠═5eb4cfb0-24c4-11eb-0829-679ddd8141f1
# ╟─9fbff700-24c4-11eb-0e18-4f2cac198d02
# ╟─e57dd0ee-24c4-11eb-20c2-ffbbfd17924b
# ╠═a7fba94e-24c4-11eb-08bd-fd17f911adfd
# ╟─18bb2fd0-24c5-11eb-17d1-2136b7554f54
# ╠═1deadaa0-24c5-11eb-0a30-fb33052e81f3
# ╟─7165b420-24c5-11eb-202c-55d63b9c9d68
# ╠═78a83960-24c5-11eb-0a79-7fc56fbb20e0
# ╟─c1a6ead2-24e3-11eb-060d-e3998d02cf74
# ╠═c8cac480-24e3-11eb-1cb3-87ad5ad98e90
# ╟─ce72857e-24e3-11eb-0821-b190f3f5239a
# ╠═d423bc60-24e3-11eb-2874-71be233234ee
# ╟─ea287c30-24e3-11eb-0338-c3ca0858b598
# ╠═ec2efa90-24e3-11eb-1275-2d2d836f93f5
# ╟─f8987810-24e3-11eb-21e9-1571dec657fd
# ╠═0a878890-24e4-11eb-19ff-1994dda9d79a
# ╠═14ea60a0-24e4-11eb-24fe-51cbe2923f63
# ╠═2a0b9e40-24e4-11eb-267b-6fb950873ceb
# ╠═31bdfc52-24e4-11eb-0868-6d2f5133b5f5
# ╠═6de045ce-24e4-11eb-1f46-ef1aa2251558
# ╟─f7e75b10-24e4-11eb-2229-9b95f98bb39e
# ╠═27d62b2e-24e5-11eb-27a7-892a41315676
# ╟─266da510-24e6-11eb-2287-1b013023b0a6
# ╠═37773790-24e6-11eb-2c64-b7218251937a
# ╠═6f6c98e0-24e9-11eb-0cf2-710690aef25b
# ╟─e1da4490-24e9-11eb-14cd-7d6890115ab8
# ╟─e352dad0-24e9-11eb-1c1d-1d900aa88432
# ╠═ea9a6920-24e9-11eb-1612-9fbb5688b457
# ╟─282ff1b0-24ea-11eb-3386-29c8d6df88fb
# ╠═2974f4d0-24ea-11eb-0b28-610bd4a077d4
# ╟─3b1d5f60-24ea-11eb-1e8c-b14da03c54ef
