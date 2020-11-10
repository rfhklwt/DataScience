### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 9e819ba0-22f8-11eb-046a-bbccee7eb7bf
begin
	using BenchmarkTools
	using DataFrames
	using DelimitedFiles
	using CSV
	using XLSX
end

# ╔═╡ ab037160-231f-11eb-3a26-df79c27f4be5
begin
	using JLD
	jld_data = JLD.load("data/mytempdata.jld")
	#save("data/mywrite.jld", "A", jld_data)
end

# ╔═╡ a531ba30-231f-11eb-167b-21562dc46128
begin
	using NPZ
	npz_data = npzread("data/mytempdata.npz")
	#npzwrite("data/mywrite.npz", npz_data)
end

# ╔═╡ 952d1f20-2320-11eb-0082-cf8e57b25ced
begin
	using MAT
	Matlab_data = matread("data/mytempdata.mat")
	#matwrite("mywrite.mat", Matlab_data)
end

# ╔═╡ 74b42950-22f8-11eb-0cb2-51110bdef4d8
md"""
## Data
Being able to easily load and process data is a crucial task that can make any data science more pleasant. In this notebook, we will cover most common types often encountered in data science tasks, and we will be using this data throughout the rest of this tutorial.
"""

# ╔═╡ edce0db0-22f8-11eb-3fa8-618c325b9d61
md"""
## 🗃️ Get some data
In Julia, it's pretty easy to dowload a file from the web using the `download` function. But also, you can use your favorite command line commad to download files by easily switching from Julia via the ; key. Let's try both.

**Note**: `download` depends on external tools such as `curl`, `wget` or `fetch`. So you must have one of these.
"""

# ╔═╡ d4631110-22fb-11eb-0a47-4da5793ecc39
download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv",
    "data/programming_languages.csv")

# ╔═╡ 6fc0d1f0-22fd-11eb-2431-8f8342ce044a
md"Another way would be to use a shell command to get the same file."

# ╔═╡ af00fd40-22fd-11eb-3696-f11c7fff4fe5
md"""
## 📂 Read your data from text files
The key question here is to load data from files such as `csv` files, `xlsx` files, or just raw text files. We will go over some Julia packages that will allow us to read such files very easily.

Let's start with the package `DelimitedFiles` which is in the standard library.
"""

# ╔═╡ b3fc9fc0-22fd-11eb-2c02-35befa9a1ab2
P, H  = readdlm("data/programming_languages.csv", ','; header=true)

# ╔═╡ 24b7e080-22fe-11eb-2d48-4f23da84fc5e
md"To write to a text file, you can:"

# ╔═╡ e822b0f0-22fd-11eb-0640-93e047f06816
#writedlm("programming_languages_dlm.txt", P, '-')

# ╔═╡ 344d97b0-22fe-11eb-3168-c5ccf8c681bb
md"""
👉 A more powerful package to use here is the `CSV` package. By default, the CSV package imports the data to a DataFrame, which can have several advantages as we will see below.

In general, `CSV.jl` is the recommended way to load CSVs in Julia. Only use `DelimitedFiles` when you have a more complicated file where you want to specify several things.
"""

# ╔═╡ 9a20f1e0-22fe-11eb-14d6-65e1a551d5d5
C = CSV.read("data/programming_languages.csv");

# ╔═╡ c48a5610-22fe-11eb-1859-e54d86987ecb
[typeof(C) typeof(P)]

# ╔═╡ b8dd98e0-22fe-11eb-12d2-a5727118933d
# C[1: 10, :]
C[!, :year] == C.year

# ╔═╡ ff25b170-22fe-11eb-1564-73c820a1fa88
P[1: 10, :]

# ╔═╡ 18389ba0-22ff-11eb-19f9-736d03e49db8
# names(C)
describe(C)

# ╔═╡ 3d8c1e90-22ff-11eb-3393-fbe47b60f1f8
@benchmark readdlm("data/programming_languages.csv",','; header=true)

# ╔═╡ 58305720-22ff-11eb-105b-e1d93f4a124f
@benchmark CSV.read("data/programming_languages.csv")

# ╔═╡ baca5840-22ff-11eb-17a1-3b68dcf0c427
md"To write to a *.csv file using the `CSV` package"

# ╔═╡ b153b4ee-22ff-11eb-318f-9d87e8d6aec4
CSV.write("data/programminglanguages_CSV.csv", DataFrame(P))

# ╔═╡ cddcda1e-22ff-11eb-36e3-958f3aaa5161
md"👉 Another type of files that we may often need to read is `XLSX` files. Let's try to read a new file."

# ╔═╡ e1cb96f0-231a-11eb-359a-7924228e0670
T = XLSX.readdata("data/zillow_data_download_april2020.xlsx", # file name
	"Sale_counts_city", # sheet name
	"A1:F9" # cell range
	)

# ╔═╡ 78d8b7d0-231b-11eb-0259-b9b354962301
md"💧 If you don't want to specify cell ranges... though this will take a little longer..."

# ╔═╡ 9c75d830-231b-11eb-022e-0d448e0ad3c1
G = XLSX.readtable("data/zillow_data_download_april2020.xlsx", "Sale_counts_city")

# ╔═╡ d96c88b0-231b-11eb-0ae0-23c558c48215
md"""
Here, `G` is a tuple of two items. The first is an vector of vectors where each vector corresponds to a column in the excel file. And the second is the header with the column names.

And we can easily store this data in a DataFrame. `DataFrame(G...)` uses the "splat" operator to _unwrap_ these arrays and pass them to the DataFrame constructor.
"""

# ╔═╡ 012042c0-231c-11eb-1300-bd01c489ea7c
D = DataFrame(G...) # equivalent to DataFrame(G[1],G[2])

# ╔═╡ 575448d0-231c-11eb-384a-398cf60123ee
begin
	foods = ["apple", "cucumber", "tomato", "banana"]
	calories = [105, 47, 22, 105]
	prices = [0.85, 1.6, 0.8, 0.6]
	dataframe_calories = DataFrame(item=foods, calories=calories)
	dataframe_prices = DataFrame(item=foods, prices=prices)
end

# ╔═╡ ccf0ed50-231c-11eb-377f-fd345a50549b
DF = innerjoin(dataframe_calories, dataframe_prices, on=:item)

# ╔═╡ e78d0f90-231c-11eb-06f8-cb8f62ee2a97
# we can also use the DataFrame constructor on a Matrix
DataFrame(T)

# ╔═╡ 282ec6b0-231d-11eb-28c3-b1f6ab9ecc26
md"""
👉 You can also easily write data to an `XLSX` file.

💧 if you already have a dataframe: 
```julia
XLSX.writetable("filename.xlsx", collect(DataFrames.eachcol(df)), DataFrames.names(df))
```
"""

# ╔═╡ 0f1748a0-231d-11eb-34a9-1100b04b9f30
#XLSX.writetable("data/writefile_using_XLSX.xlsx", G[1], G[2])

# ╔═╡ cfb871c0-231c-11eb-2714-27cd9758de2c
md"""
## ⬇️ Importing your data

Often, the data you want to import is not stored in plain text, and you might want to import different kinds of types. Here we will go over importing `jld`, `npz`, `rda`, and `mat` files. Hopefully, these four will capture the types from four common programming languages used in Data Science (Julia, Python, R, Matlab).

We will use a toy example here of a very small matrix. But the same syntax will hold for bigger files.

```julia
4×5 Array{Int64,2}:
 2  1446  1705  1795  1890
 3  2926  3121  3220  3405
 4  2910  3022  2937  3224
 5  1479  1529  1582  1761
```
"""

# ╔═╡ 2f418200-2320-11eb-0a97-0341c70b5d82
# begin
# 	using RData
# 	R_data = RData.load("data/mytempdata.rda")
# 	# We'll need RCall to save here. https://github.com/JuliaData/RData.jl/issues/56
# 	using RCall
# 	@rput R_data
# 	R"save(R_data, file=\"mywrite.rda\")"
# end

# ╔═╡ 057838f0-2321-11eb-1c33-43614a1f3512
["jld_data" => typeof(jld_data), "npz_data" => typeof(npz_data), "Matlab_data" => typeof(Matlab_data)]

# ╔═╡ 593d13c0-2321-11eb-3e17-7fd0fa7393be
Matlab_data

# ╔═╡ 5d9e9920-2321-11eb-0672-137bc72bf655
md"""
# 🔢 Time to process the data from Julia
We will mainly cover `Matrix` (or `Vector`), `DataFrame`s, and `dict`s (or dictionaries). Let's bring back our programming languages dataset and start playing it the matrix it's stored in.
"""

# ╔═╡ 7accb0e0-2321-11eb-3f6a-dd948bf0b00e
P

# ╔═╡ 90dd7ea0-2321-11eb-24be-bdc01aa487a8
md"""
Here are some quick questions we might want to ask about this simple data.
- Which year was was a given language invented?
- How many languages were created in a given year?
"""

# ╔═╡ 94a498c0-2321-11eb-3094-bf0f38b61660
# Q1: Which year was was a given language invented?
function year_created(P, language::String)
	loc = findfirst(P[:, 2] .== language)
	
	P[loc, 1]
end

# ╔═╡ f5bf9b02-2321-11eb-2c39-b1873c26cb25
function year_created_handle_error(P, language::String)
	loc = findfirst(P[:, 2] .== language)
	!isnothing(loc) && return P[loc, 1]
	error("Error: Language not found.")
end

# ╔═╡ 3eaeb7b0-2322-11eb-329d-7d3486827f08
# Q2: How many languages were created in a given year?
how_many_per_year(P, year::Int64) = length(findall(P[:, 1] .== year))

# ╔═╡ 894e91a0-2322-11eb-31a9-b30bb0196e70
md"👇 Now let's try to store this data in a DataFrame..."

# ╔═╡ 950f4c00-2322-11eb-132e-f1b68f84f9ee
P_df = C #DataFrame(year = P[:, 1], language = P[:, 2]) # or DataFrame(P)

# ╔═╡ 810ddca0-232a-11eb-0da2-834a192fe9b9
md"Even better, since we know the types of each column, we can create the DataFrame as follows:"

# ╔═╡ 91d276e0-232a-11eb-1405-ab8fd8a66935
P_df2 = DataFrame(year = Int.(P[:, 1]), language = string.(P[:, 2]))

# ╔═╡ afc56540-232a-11eb-06de-775bb9c3a9f9
md"👇 And now let's answer the same questions we just answered..."

# ╔═╡ c38eb400-232a-11eb-08be-1559c25ac17a
# Q1: Which year was was a given language invented?
# it's a little more intuitive and you don't need to remember the column ids
function year_created(P_df::DataFrame, language::String)
	loc = findfirst(P_df.language .== language)
	
	P_df.year[loc]
end

# ╔═╡ 6a416a90-232b-11eb-0acf-673a04173087
function year_created_handle_error(P_df::DataFrame,language::String)
    loc = findfirst(P_df.language .== language)
    !isnothing(loc) && return P_df.year[loc]
    
	error("Error: Language not found.")
end

# ╔═╡ 398c7a62-2322-11eb-073f-91d939e9b158
year_created_handle_error(P, "W")

# ╔═╡ 782be2c0-232b-11eb-204f-1f799bf3b627
year_created_handle_error(P_df, "W")

# ╔═╡ 7b61b960-232b-11eb-049e-5defd3c4e43d
# Q2: How many languages were created in a given year?
how_many_per_year(P_df::DataFrame, year::Int64) = length(findall(P_df.year .== year))

# ╔═╡ a866fe20-232b-11eb-033a-6965a847af24
md"🔥 Next, we'll use dictionaries. A quick way to create a dictionary is with the `Dict()` command. But this creates a dictionary without types. Here, we will specify the types of this dictionary."

# ╔═╡ 2666b310-232c-11eb-01b9-bda9f9df7a93
md"""
```julia
# A quick example to show how to build a dictionary
julia> Dict([("A", 1), ["B", 2], [1, [1, 2]]])
Dict{Any,Any} with 3 entries:
  "B" => 2
  "A" => 1
  1   => [1, 2]

julia> P_dictionary = Dict{Integer, Vector{String}}()
Dict{Integer,Array{String,1}} with 0 entries

julia> P_dictionary[67] = ["julia","programming"]
2-element Array{String,1}:
 "julia"
 "programming"

# this is not gonna work.
julia> P_dictionary["julia"] = 7
ERROR: MethodError: Cannot `convert` an object of type String to an object of type Integer
```
"""

# ╔═╡ e2e984e2-232c-11eb-16ea-312934108e0c
md"👉 Now, let's populate the dictionary with years as keys and vectors that hold all the programming languages created in each year as their values. Even though this looks like more work, we often need to do it just once."

# ╔═╡ f5afbb80-232c-11eb-26f3-ff2c974f6988
begin
	dict = Dict{Integer, Vector{String}}()
	for i in 1: size(P, 1)
		year, lang = P[i, :]
		if year in keys(dict)
			dict[year] = push!(dict[year], lang)
			# note that push! is not our favorite thing to do in Julia, 
        	# but we're focusing on correctness rather than speed here
		else
			dict[year] = [lang]
		end
	end
end

# ╔═╡ 0b4f3460-232e-11eb-1894-3fb6b8fa2776
md"👇 Though a smarter way to do this is:"

# ╔═╡ 10ff80e0-232e-11eb-35e1-5f568607f9bb
begin
	P_dictionary = Dict{Integer, Vector{String}}()
	
	curyear = P_df.year[1]
	P_dictionary[curyear] = [P_df.language[1]]
	
	for (i, nextyear) in enumerate(P_df.year[2: end])
		global curyear
		if nextyear == curyear
			# same key
			P_dictionary[curyear] = push!(P_dictionary[curyear], P_df.language[i + 1])
		else
			curyear = nextyear
			P_dictionary[curyear] = [P_df.language[i + 1]]
		end
	end
end

# ╔═╡ 0b8f38b0-2330-11eb-1e64-9f4e3eb639d0
typeof(P_dictionary) <: Dict

# ╔═╡ e40654a0-232e-11eb-1940-497ece2348cf
length(keys(P_dictionary)) == length(unique(P[:,1]))

# ╔═╡ 8363ac20-232d-11eb-287b-49488b92fd3e
# Q1: Which year was was a given language invented?
# now instead of looking in one long vector, we will look in many small vectors
function year_created(P_dictionary::Dict, language::String)
	keys_vec = collect(keys(P_dictionary))
	lookup = map(keyid -> findfirst(P_dictionary[keyid] .== language), keys_vec)
	
	keys_vec[findfirst((!isnothing).(lookup))]
end

# ╔═╡ ec7b2b40-2321-11eb-0f4a-616afbd55395
year_created(P, "Julia")

# ╔═╡ 10b056d0-232b-11eb-34ef-c575515be077
year_created(P_df, "Julia")

# ╔═╡ 28cd5600-2330-11eb-13ba-c37616f87d01
year_created(P_dictionary, "Julia")

# ╔═╡ 08c8b3c0-2332-11eb-2428-e10efee2b691
# Q2: How many languages were created in a given year?
how_many_per_year(P_dictionary::Dict, year::Int64) = length(P_dictionary[year])

# ╔═╡ 7958e01e-2322-11eb-35b3-1d8783967831
how_many_per_year(P, 2011)

# ╔═╡ a43932f0-232b-11eb-0ef7-bd1c503e4059
how_many_per_year(P_df, 2011)

# ╔═╡ 27a6ec80-2332-11eb-28f1-a355a1b52e72
how_many_per_year(P_dictionary, 2011)

# ╔═╡ 2e5017f0-2332-11eb-2d0d-19ceed2d9714
md"## 📝 A note about missing data"

# ╔═╡ 3b46e150-2332-11eb-22b9-b3a5a7770fd5
# assume there were missing values in our dataframe
begin
	P[1, 1] = missing
	P_df3 = DataFrame(year = P[:, 1], language = P[:, 2])
end

# ╔═╡ 6bc8a2a0-2332-11eb-3514-076e86a69746
dropmissing(P_df3)

# ╔═╡ 9e5994e0-2332-11eb-0566-ef42ddba9dd2
md"""
## Finally...

After finishing this notebook, you should be able to:
- [ ] dowload a data file from the web given a url
- [ ] load data from a file from a text file via DelimitedFiles or CSV
- [ ] write your data to a text file or csv file
- [ ] load data from file types xlsx, jld, npz, mat, rda
- [ ] write your data to an xlsx file, jld, npz, mat, rda
- [ ] store data in a 2D array (`Matrix`), or `DataFrame` or `Dict`
- [ ] write functions to perform basic lookups on `Matrix`, `DataFrame`, and `Dict` types
- [ ] use some of the basic functions on `DataFrame`s such as: `dropmissing`, `describe`, `by`, and `join`
"""

# ╔═╡ Cell order:
# ╟─74b42950-22f8-11eb-0cb2-51110bdef4d8
# ╠═9e819ba0-22f8-11eb-046a-bbccee7eb7bf
# ╟─edce0db0-22f8-11eb-3fa8-618c325b9d61
# ╠═d4631110-22fb-11eb-0a47-4da5793ecc39
# ╟─6fc0d1f0-22fd-11eb-2431-8f8342ce044a
# ╟─af00fd40-22fd-11eb-3696-f11c7fff4fe5
# ╠═b3fc9fc0-22fd-11eb-2c02-35befa9a1ab2
# ╟─24b7e080-22fe-11eb-2d48-4f23da84fc5e
# ╠═e822b0f0-22fd-11eb-0640-93e047f06816
# ╟─344d97b0-22fe-11eb-3168-c5ccf8c681bb
# ╠═9a20f1e0-22fe-11eb-14d6-65e1a551d5d5
# ╠═c48a5610-22fe-11eb-1859-e54d86987ecb
# ╠═b8dd98e0-22fe-11eb-12d2-a5727118933d
# ╠═ff25b170-22fe-11eb-1564-73c820a1fa88
# ╠═18389ba0-22ff-11eb-19f9-736d03e49db8
# ╠═3d8c1e90-22ff-11eb-3393-fbe47b60f1f8
# ╠═58305720-22ff-11eb-105b-e1d93f4a124f
# ╟─baca5840-22ff-11eb-17a1-3b68dcf0c427
# ╠═b153b4ee-22ff-11eb-318f-9d87e8d6aec4
# ╟─cddcda1e-22ff-11eb-36e3-958f3aaa5161
# ╠═e1cb96f0-231a-11eb-359a-7924228e0670
# ╟─78d8b7d0-231b-11eb-0259-b9b354962301
# ╠═9c75d830-231b-11eb-022e-0d448e0ad3c1
# ╟─d96c88b0-231b-11eb-0ae0-23c558c48215
# ╠═012042c0-231c-11eb-1300-bd01c489ea7c
# ╠═575448d0-231c-11eb-384a-398cf60123ee
# ╠═ccf0ed50-231c-11eb-377f-fd345a50549b
# ╠═e78d0f90-231c-11eb-06f8-cb8f62ee2a97
# ╟─282ec6b0-231d-11eb-28c3-b1f6ab9ecc26
# ╠═0f1748a0-231d-11eb-34a9-1100b04b9f30
# ╟─cfb871c0-231c-11eb-2714-27cd9758de2c
# ╠═ab037160-231f-11eb-3a26-df79c27f4be5
# ╠═a531ba30-231f-11eb-167b-21562dc46128
# ╠═2f418200-2320-11eb-0a97-0341c70b5d82
# ╠═952d1f20-2320-11eb-0082-cf8e57b25ced
# ╟─057838f0-2321-11eb-1c33-43614a1f3512
# ╠═593d13c0-2321-11eb-3e17-7fd0fa7393be
# ╟─5d9e9920-2321-11eb-0672-137bc72bf655
# ╠═7accb0e0-2321-11eb-3f6a-dd948bf0b00e
# ╟─90dd7ea0-2321-11eb-24be-bdc01aa487a8
# ╠═94a498c0-2321-11eb-3094-bf0f38b61660
# ╠═ec7b2b40-2321-11eb-0f4a-616afbd55395
# ╠═f5bf9b02-2321-11eb-2c39-b1873c26cb25
# ╠═398c7a62-2322-11eb-073f-91d939e9b158
# ╠═3eaeb7b0-2322-11eb-329d-7d3486827f08
# ╠═7958e01e-2322-11eb-35b3-1d8783967831
# ╟─894e91a0-2322-11eb-31a9-b30bb0196e70
# ╠═950f4c00-2322-11eb-132e-f1b68f84f9ee
# ╟─810ddca0-232a-11eb-0da2-834a192fe9b9
# ╠═91d276e0-232a-11eb-1405-ab8fd8a66935
# ╟─afc56540-232a-11eb-06de-775bb9c3a9f9
# ╠═c38eb400-232a-11eb-08be-1559c25ac17a
# ╠═10b056d0-232b-11eb-34ef-c575515be077
# ╠═6a416a90-232b-11eb-0acf-673a04173087
# ╠═782be2c0-232b-11eb-204f-1f799bf3b627
# ╠═7b61b960-232b-11eb-049e-5defd3c4e43d
# ╠═a43932f0-232b-11eb-0ef7-bd1c503e4059
# ╟─a866fe20-232b-11eb-033a-6965a847af24
# ╟─2666b310-232c-11eb-01b9-bda9f9df7a93
# ╟─e2e984e2-232c-11eb-16ea-312934108e0c
# ╠═f5afbb80-232c-11eb-26f3-ff2c974f6988
# ╟─0b4f3460-232e-11eb-1894-3fb6b8fa2776
# ╠═10ff80e0-232e-11eb-35e1-5f568607f9bb
# ╠═0b8f38b0-2330-11eb-1e64-9f4e3eb639d0
# ╠═e40654a0-232e-11eb-1940-497ece2348cf
# ╠═8363ac20-232d-11eb-287b-49488b92fd3e
# ╠═28cd5600-2330-11eb-13ba-c37616f87d01
# ╠═08c8b3c0-2332-11eb-2428-e10efee2b691
# ╠═27a6ec80-2332-11eb-28f1-a355a1b52e72
# ╟─2e5017f0-2332-11eb-2d0d-19ceed2d9714
# ╠═3b46e150-2332-11eb-22b9-b3a5a7770fd5
# ╠═6bc8a2a0-2332-11eb-3514-076e86a69746
# ╟─9e5994e0-2332-11eb-0566-ef42ddba9dd2
