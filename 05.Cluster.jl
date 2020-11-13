### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 00f0c3fe-24ed-11eb-34bb-db8f13e8f1c3
begin
	# Packages we will use throughout this notebook
	using Clustering
	using VegaLite
	using VegaDatasets
	using DataFrames
	using Statistics
	using JSON
	using CSV
	using Distances
end

# ╔═╡ e4314c90-24ec-11eb-31fb-8b04564f1a67
md"""
## Clustering
Put simply, the task of clustering is to place observations that seem similar within the same cluster. Clustering is commonly used in two dimensional data where the goal is to create clusters based on coordinates. Here, we will use something similar. We will cluster houses based on their latitude-longitude locations using several different clustering methods.
"""

# ╔═╡ 5cc70cd0-24ed-11eb-28a7-43a742fd0d7f
md"We will start off by getting some data. We will use data of 20,000+ California houses dataset. We will then learn whether housing prices directly correlate with map location."

# ╔═╡ a3725400-24ed-11eb-0bff-29c57455ae40
begin
	download("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv", "newhouses.csv")
	houses = CSV.read("newhouses.csv")
end

# ╔═╡ 4df0dfa0-24ee-11eb-3e7b-b133de452fa7
names(houses)

# ╔═╡ 288b9610-2575-11eb-397e-a9ffbb6de598
md"""
We will use the `VegaLite` package here for plotting. This package makes it very easy to plot information on a map. All you need is a JSON file of the map you intend to draw. Here, we will use the California counties JSON file and plot each house on the map and color code it via a heatmap of the price. This is done by this line `color="median_house_value:q"`
"""

# ╔═╡ daa38ea0-2577-11eb-3535-191e383d8d5a
begin
	cali_shape = JSON.parsefile("data/california-counties.json")
	VV = VegaDatasets.VegaJSONDataset(cali_shape,"data/california-counties.json")
end

# ╔═╡ 2f9ea6e0-2575-11eb-0bea-51851d442dcc
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,
			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	) +
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="median_house_value:q"
	)
end

# ╔═╡ 2acc8560-2575-11eb-3e0c-fbdaa104f769
bucketprice = Int.(houses[!,:median_house_value] .÷ 50000)

# ╔═╡ ff6db2c0-2576-11eb-182c-0bca43b8976f
insertcols!(houses, 3, :cprice=>bucketprice)

# ╔═╡ 3374f150-2577-11eb-126d-cf972a3748cb
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,
			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	)+
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="cprice:n"

	)
end

# ╔═╡ 256659de-2578-11eb-3de7-bd23ef913a1d
md"### 🟤K-means clustering"

# ╔═╡ 4e56d9b0-2578-11eb-3594-0982631d7c1a
begin
	X = houses[!, [:latitude, :longitude]]
	C = kmeans(Matrix(X)', 10)
	insertcols!(houses, 3, :cluster10=>C.assignments)
end

# ╔═╡ fae75560-2578-11eb-1756-0d553ad1e5bf
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,
			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	)+
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="cluster10:n"
	)
end

# ╔═╡ bac3f600-2578-11eb-1144-5f5646da05dd
md"Yes, location affects price of the house but this means location as in proximity to water, prosimity to downtown, promisity to a bus stop and so on

lets' see if this remains true for the rest."

# ╔═╡ eaee4120-257b-11eb-3f5f-23daca4524fc
md"""
### 🟤K-medoids clustering
For this type of clustering, we need to build a distance matrix. We will use the `Distances` package for this purpose and compute the pairwise Euclidean distances.
"""

# ╔═╡ f4d161e0-257b-11eb-1fb0-9577ed6c82ee
begin
	xmatrix = Matrix(X)'
	D = pairwise(Euclidean(), xmatrix, xmatrix, dims=2)
	
	K = kmedoids(D, 10)
	insertcols!(houses, 3, :medoids_clusters=>K.assignments)
end

# ╔═╡ 1afacb10-257f-11eb-3ca3-f153e1d997aa
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,
			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	)+
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="medoids_clusters:n"

	)
end

# ╔═╡ 24753ef0-257f-11eb-003c-a76d68cfef72
md"### 🟤Hierarchial Clustering"

# ╔═╡ 5ec85140-2580-11eb-36de-69bb2361eb46
begin
	H = hclust(D)
	L = cutree(H; k=10)
	insertcols!(houses, 3, :hclust_clusters=>L)
end

# ╔═╡ b309fd30-2580-11eb-3a4f-a5fcb69d10c2
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,
			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	)+
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="hclust_clusters:n"
	)
end

# ╔═╡ dc74dd1e-2580-11eb-054f-43913d29cf98
md"### 🟤DBscan"

# ╔═╡ e4e90490-2580-11eb-0bca-dd1126f49bf9
begin
	dclara = pairwise(SqEuclidean(), Matrix(X)', dims=2)
	clusters = dbscan(dclara, 0.05, 10)
	insertcols!(houses, 3, :dbscan_clusters=>clusters.assignments)
end

# ╔═╡ 0678d34e-2582-11eb-2728-29cef28fc6cc
let
	@vlplot(width=500, height=300) +
	@vlplot(
		mark={
			:geoshape,

			fill=:black,
			stroke=:white
		},
		data={
			values=VV,
			format={
				type=:topojson,
				feature=:cb_2015_california_county_20m
			}
		},
		projection={type=:albersUsa},
	)+
	@vlplot(
		:circle,
		data=houses,
		projection={type=:albersUsa},
		longitude="longitude:q",
		latitude="latitude:q",
		size={value=12},
		color="dbscan_clusters:n"

	)
end

# ╔═╡ 1a0d7d80-2582-11eb-11fe-137751c18c60
md"""
# Finally...
After finishing this notebook, you should be able to:
- [ ] run kmeans clustering on your data
- [ ] run kmedoids clustering on your data
- [ ] run hierarchial clustering on your data
- [ ] run DBscan clustering on your data
- [ ] modify a dataframe and add a new named column
- [ ] generate good looking plots of maps using the VegaLite package
"""

# ╔═╡ 1ccff8e0-2582-11eb-0249-c136aca6ef14
md"""
# 🥳 One cool finding

Prices in California do not seem to have an exact mapping with geographical locations. In specifc, performing a clustering algorithm on the houses dataset we had did not reveal a mapping with the price ranges. This indicate that prices relationship to geographical location is not necessairly based on neighborhood but probably other factors like closeness to the water or closeness to a downtown. Here is a figure with a heat map of prices 
![](https://raw.githubusercontent.com/rfhklwt/DataScience/master/data/0501.png)
And here is a and k-means clustering of the same houses based on their location
![](https://raw.githubusercontent.com/rfhklwt/DataScience/master/data/0502.png)

"""

# ╔═╡ Cell order:
# ╟─e4314c90-24ec-11eb-31fb-8b04564f1a67
# ╠═00f0c3fe-24ed-11eb-34bb-db8f13e8f1c3
# ╟─5cc70cd0-24ed-11eb-28a7-43a742fd0d7f
# ╠═a3725400-24ed-11eb-0bff-29c57455ae40
# ╠═4df0dfa0-24ee-11eb-3e7b-b133de452fa7
# ╟─288b9610-2575-11eb-397e-a9ffbb6de598
# ╠═daa38ea0-2577-11eb-3535-191e383d8d5a
# ╠═2f9ea6e0-2575-11eb-0bea-51851d442dcc
# ╠═2acc8560-2575-11eb-3e0c-fbdaa104f769
# ╠═ff6db2c0-2576-11eb-182c-0bca43b8976f
# ╠═3374f150-2577-11eb-126d-cf972a3748cb
# ╟─256659de-2578-11eb-3de7-bd23ef913a1d
# ╠═4e56d9b0-2578-11eb-3594-0982631d7c1a
# ╠═fae75560-2578-11eb-1756-0d553ad1e5bf
# ╟─bac3f600-2578-11eb-1144-5f5646da05dd
# ╟─eaee4120-257b-11eb-3f5f-23daca4524fc
# ╠═f4d161e0-257b-11eb-1fb0-9577ed6c82ee
# ╠═1afacb10-257f-11eb-3ca3-f153e1d997aa
# ╠═24753ef0-257f-11eb-003c-a76d68cfef72
# ╠═5ec85140-2580-11eb-36de-69bb2361eb46
# ╠═b309fd30-2580-11eb-3a4f-a5fcb69d10c2
# ╟─dc74dd1e-2580-11eb-054f-43913d29cf98
# ╠═e4e90490-2580-11eb-0bca-dd1126f49bf9
# ╠═0678d34e-2582-11eb-2728-29cef28fc6cc
# ╠═1a0d7d80-2582-11eb-11fe-137751c18c60
# ╟─1ccff8e0-2582-11eb-0249-c136aca6ef14
