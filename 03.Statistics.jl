### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 741484b0-23fa-11eb-1b0e-6f1c1c3f5de5
begin
	using Statistics
	using StatsBase
	using RDatasets
	using Plots
	using StatsPlots
	using KernelDensity
	using Distributions
	using LinearAlgebra
	using HypothesisTests
	using PyCall
	using MLBase
	using SciPy
end

# ╔═╡ aec24c10-23f9-11eb-0a86-cfdc6f354641
md"""
## Statistics
Having a solid understanding of statistics in data science allows us to understand our data better, and allows us to create a quantifiable evaluation of any future conclusions.
"""

# ╔═╡ 4840506e-23fb-11eb-08fb-a386f1bf7d43
md"""
🌕 In this notebook, we will use eruption data on the faithful geyser. The data will contain wait times between every consecutive times the geyser goes off and the length of the eruptions.
![](https://github.com/rfhklwt/DataScience/blob/master/data/faithful.JPG?raw=true)
"""

# ╔═╡ 96372820-23fc-11eb-052b-e3ef2d328341
md"Let's get the data first..."

# ╔═╡ 05decfd2-241a-11eb-0cec-c36e87fba6c9
D = dataset("datasets","faithful")

# ╔═╡ 466277a0-241a-11eb-190d-99fe7f9426ba
names(D)

# ╔═╡ 4e1a05d2-241a-11eb-3588-73a9bf5a971c
describe(D)

# ╔═╡ 53637a30-241a-11eb-06b3-31a8167611d6
begin
	eruptions = D[!, :Eruptions]
	scatter(eruptions, label="eruptions")
	waittime = D[!, :Waiting]
	scatter!(waittime, label="wait time")
end

# ╔═╡ cddb19d0-241a-11eb-0079-f7b68976c3a9
md"""
### 🔵Statistics plots
As you can see, this doesn't tell us much about the data... Let's try some statistical plots
"""

# ╔═╡ 17c9d5c0-2427-11eb-0a42-bf44d3a05395
boxplot(["eruption length"], eruptions, legend=false, size=(200, 400), wisker_width=1, ylabel="time in minutes")

# ╔═╡ f6973960-24ad-11eb-0b70-79a5736dd58f
md"""
Statistical plots such as a box plot (and a violin plot as we will see in notebook `12. Visualization`), can provide a much better understanding of the data. Here, we immediately see that the median time of each eruption is about 4 minutes.

The next plot we will see is a histogram plot.
"""

# ╔═╡ 375b93b0-24ae-11eb-29ca-5d8153cc4952
histogram(eruptions, label="eruptions")

# ╔═╡ 48dea680-24af-11eb-21cc-e5209deab0ab
md"☝️ You can adjust the number of bins manually or by passing a one of the autobinning functions."

# ╔═╡ 5c1cf350-24af-11eb-3542-41d59d641c10
histogram(eruptions, bins=:sqrt, label="eruptions")

# ╔═╡ a96d202e-24af-11eb-3778-b592d55183b4
md"""
### 🔵Kernel density estimates
Next, we will see how we can fit a kernel density estimation function to our data. We will make use of the `KernelDensity.jl` package. 
"""

# ╔═╡ 011ca2ae-24b0-11eb-0d4a-2149c923de08
p = kde(eruptions)

# ╔═╡ 0c38b8a0-24b0-11eb-0f51-8ba70d276ec1
md"""
If we want the histogram and the kernel density graph to be aligned we need to remember that the \"density contribution\" of every point added to one of these histograms is `1/(nb of elements)*bin width`.

💬 Read more about kernel density estimates on its [wikipedia page](https://en.wikipedia.org/wiki/Kernel_density_estimation)
"""

# ╔═╡ 9097227e-24b0-11eb-201c-7b4a3a37bae1
begin
	histogram(eruptions, label="eruptions")
	# number of elements * binwidth
	plot!(p.x, p.density .* length(eruptions), linewidth=3, color=2, label="kde fit")
end

# ╔═╡ c37dfcf0-24b0-11eb-3716-1586085370f3
begin
	histogram(eruptions, bins=:sqrt, label="eruptions")
	# number of elements * binwidth
	plot!(p.x, p.density .* length(eruptions) .* 0.2, linewidth=3, color=2, label="kde fit")
end

# ╔═╡ e2254730-24b0-11eb-03d6-0b2b810b55ca
md"Next, we will take a look at one probablity distribution, namely the normal distribution and verify that it generates a bell curve."

# ╔═╡ e5917010-24b0-11eb-10fe-2faa8330079c
let
	random_vector = randn(100_000)
	histogram(random_vector)
	p = kde(random_vector)
	plot!(p.x, p.density * length(random_vector) .* 0.1,
		  linewidth=3, color=2, label="kde fit")
end

# ╔═╡ 27597ec0-24b1-11eb-0c65-99642a3ba044
md"""
### 🔵Probability distributions
Another way to generate the same plot is via using the `Distributions` package and choosing the probability distribution you want, and then drawing random numbers from it. As an example, we will use `d = Normal()` below.
"""

# ╔═╡ 388a2140-24b1-11eb-2230-dd6579382528
let
	d = Normal()
	rand_vector = rand(d, 100_000)
	histogram(rand_vector)
	p = kde(rand_vector)
	plot!(p.x, p.density .* length(rand_vector) .* 0.1,
		  linewidth=3, color=2, label="kde fit")
end

# ╔═╡ 9200a820-24b1-11eb-1ba2-11c254e4be15
let
	b = Binomial(40)
	rand_vector = rand(b, 100_000)
	histogram(rand_vector)
	p = kde(rand_vector)
	plot!(p.x, p.density .* length(rand_vector) .* 0.5,
		  linewidth=3, color=2, label="kde fit")
end

# ╔═╡ 311795e0-24b2-11eb-2447-73fb48c76dba
md"Next, we will try to fit a given set of numbers to a distribution."

# ╔═╡ 33be4be0-24b2-11eb-2e5b-cd67e17f93ee
let
	x = rand(100_0)
	d = fit(Normal, x)
	rand_vector = rand(d, 100_0)
	histogram(rand_vector, nbins=20, fillalpha=0.3, label="fit")
	histogram!(x, nbins=20, linecolor=:red, fillalpha=0.3, label="myvector")
end

# ╔═╡ ff7071f0-24b2-11eb-0166-5bbce0e747f3
let
	x = eruptions
	d = fit(Normal, x)
	rand_vector = rand(d, 100_0)
	histogram(rand_vector, nbins=20, fillalpha=0.3, label="fit")
	histogram!(x, nbins=20, linecolor=:red, fillalpha=0.3, label="myvector")
end

# ╔═╡ 1af93e20-24b3-11eb-3ec2-01a06ed64eeb
md"""
### 🔵Hypothesis testing
Next, we will perform hypothesis testing using the `HypothesisTests.jl` package.
"""

# ╔═╡ 22da9b70-24b3-11eb-1739-b1a2b8873eff
let
	rand_vector = randn(100_0)
	OneSampleTTest(rand_vector)
end

# ╔═╡ 798de3a0-24b3-11eb-23e7-cd845d9151d4
OneSampleTTest(eruptions)

# ╔═╡ a3617070-24b3-11eb-0008-e16c6d07c168
SciPy.stats.spearmanr(eruptions, waittime)

# ╔═╡ c2f98350-24b3-11eb-18b4-65fbaad57d4a
SciPy.stats.pearsonr(eruptions, waittime)

# ╔═╡ f5b0579e-24b4-11eb-2ced-91fadb6f6a06
corspearman(eruptions, waittime)

# ╔═╡ 064667d0-24b5-11eb-1061-8b2062d2269b
cor(eruptions, waittime)

# ╔═╡ 0c978830-24b5-11eb-0868-03004142462f
scatter(eruptions, waittime, xlabel="eruption length",
    ylabel="wait time between eruptions", legend=false, grid=false, size=(400,300))

# ╔═╡ 160997a0-24b5-11eb-0281-b99924bb1726
md"Interesting! This means that the next time you visit Yellowstone National part ot see the faithful geysser and you have to wait for too long for it to go off, you will likely get a longer eruption! "

# ╔═╡ 194ba340-24b5-11eb-27ea-7bbc7045e8d3
md"""
### 🔵AUC and Confusion matrix
Finally, we will cover basic tools you will need such as AUC scores or confusion matrix. We use the `MLBase` package for that.
"""

# ╔═╡ 3d0c2a20-24b5-11eb-3502-fde84a4b0de8
let
	gt = [1, 1, 1, 1, 1, 1, 1, 2]
	pred = [1, 1, 2, 2, 1, 1, 1, 1]
	C = confusmat(2, gt, pred)   # compute confusion matrix
	C ./ sum(C, dims=2)   # normalize per class
	sum(diag(C)) / length(gt)  # compute correct rate from confusion matrix
	correctrate(gt, pred)
	C = confusmat(2, gt, pred)   
end

# ╔═╡ 7f681460-24b5-11eb-3043-fb4ba1e87244
let
	gt = [1, 1, 1, 1, 1, 1, 1, 0]
	pred = [1, 1, 0, 0, 1, 1, 1, 1]
	ROC = MLBase.roc(gt,pred)
	recall(ROC)
	precision(ROC)
end

# ╔═╡ 89f69050-24b5-11eb-005e-e5ad1e1dbccf
md"""
# Finally...
After finishing this notebook, you should be able to:
- [ ] generate statistics plots such as box plot, histogram, and kernel densities
- [ ] generate distributions in Julia, and draw random numbers accordingly
- [ ] fit a given set of numbers to a distribution
- [ ] compute basic evaluation metrics such as AUC and confusion matrix
- [ ] run hypothesis testing
- [ ] compute correlations and p-values
"""

# ╔═╡ b7809570-24b5-11eb-01c6-5771da1f4c93
md"""
# 🥳 One cool finding

If you go Yellowstone national park and you find out that the old faithful geyser is taking too long to erupt, then the wait might be worth it because you are likely to experience a longer eruption (i.e. there seems to be a high correlation between wait time and eruption time).


"""

# ╔═╡ Cell order:
# ╠═aec24c10-23f9-11eb-0a86-cfdc6f354641
# ╠═741484b0-23fa-11eb-1b0e-6f1c1c3f5de5
# ╟─4840506e-23fb-11eb-08fb-a386f1bf7d43
# ╟─96372820-23fc-11eb-052b-e3ef2d328341
# ╠═05decfd2-241a-11eb-0cec-c36e87fba6c9
# ╠═466277a0-241a-11eb-190d-99fe7f9426ba
# ╠═4e1a05d2-241a-11eb-3588-73a9bf5a971c
# ╠═53637a30-241a-11eb-06b3-31a8167611d6
# ╟─cddb19d0-241a-11eb-0079-f7b68976c3a9
# ╠═17c9d5c0-2427-11eb-0a42-bf44d3a05395
# ╟─f6973960-24ad-11eb-0b70-79a5736dd58f
# ╠═375b93b0-24ae-11eb-29ca-5d8153cc4952
# ╟─48dea680-24af-11eb-21cc-e5209deab0ab
# ╠═5c1cf350-24af-11eb-3542-41d59d641c10
# ╟─a96d202e-24af-11eb-3778-b592d55183b4
# ╠═011ca2ae-24b0-11eb-0d4a-2149c923de08
# ╟─0c38b8a0-24b0-11eb-0f51-8ba70d276ec1
# ╠═9097227e-24b0-11eb-201c-7b4a3a37bae1
# ╠═c37dfcf0-24b0-11eb-3716-1586085370f3
# ╟─e2254730-24b0-11eb-03d6-0b2b810b55ca
# ╠═e5917010-24b0-11eb-10fe-2faa8330079c
# ╟─27597ec0-24b1-11eb-0c65-99642a3ba044
# ╠═388a2140-24b1-11eb-2230-dd6579382528
# ╠═9200a820-24b1-11eb-1ba2-11c254e4be15
# ╟─311795e0-24b2-11eb-2447-73fb48c76dba
# ╠═33be4be0-24b2-11eb-2e5b-cd67e17f93ee
# ╠═ff7071f0-24b2-11eb-0166-5bbce0e747f3
# ╟─1af93e20-24b3-11eb-3ec2-01a06ed64eeb
# ╠═22da9b70-24b3-11eb-1739-b1a2b8873eff
# ╠═798de3a0-24b3-11eb-23e7-cd845d9151d4
# ╠═a3617070-24b3-11eb-0008-e16c6d07c168
# ╠═c2f98350-24b3-11eb-18b4-65fbaad57d4a
# ╠═f5b0579e-24b4-11eb-2ced-91fadb6f6a06
# ╠═064667d0-24b5-11eb-1061-8b2062d2269b
# ╠═0c978830-24b5-11eb-0868-03004142462f
# ╟─160997a0-24b5-11eb-0281-b99924bb1726
# ╟─194ba340-24b5-11eb-27ea-7bbc7045e8d3
# ╠═3d0c2a20-24b5-11eb-3502-fde84a4b0de8
# ╠═7f681460-24b5-11eb-3043-fb4ba1e87244
# ╟─89f69050-24b5-11eb-005e-e5ad1e1dbccf
# ╟─b7809570-24b5-11eb-01c6-5771da1f4c93
