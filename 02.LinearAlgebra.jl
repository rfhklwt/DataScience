### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ be373c70-2357-11eb-1e3e-39b91320918a
begin
	using LinearAlgebra
	using SparseArrays
	using Images
	using MAT
end

# ╔═╡ 56cfadb2-2357-11eb-3e5c-2dc86275630e
md"""
## Linear Algebra
A lot of the Data Science methods we will see in this tutorial require some understanding of linear algebra, and in this notebook we will focus on how Julia handles matrices, the types that exist, and how to call basic linear algebra tasks.
"""

# ╔═╡ cc4a4b40-2357-11eb-2e5e-cdbb8424f76c
md"Some packages we will use: 👇"

# ╔═╡ 8e43d110-235a-11eb-3415-674c3aeb39b9
md"""
![title](https://raw.githubusercontent.com/rfhklwt/DataScience/master/data/matrix_storage.png)
"""

# ╔═╡ 579794c0-235b-11eb-1298-a7f5b27db5f8
md"""
### 🟢Getting started
We will get started with creating a random matrix.
"""

# ╔═╡ 6486c790-235c-11eb-2461-d74d43dbf622
begin
	A = rand(10, 10)  	# created a random matrix of size 10-by-10
	Atranspose = A'   	# matrix transpose
	A = A * Atranspose  # matrix multiplication
end

# ╔═╡ 8ab233f0-235c-11eb-0830-7d3f4d3590f4
A[11] == A[1, 2]

# ╔═╡ b9158210-235c-11eb-1dee-43889fe78fd6
begin
	b = rand(10)	# created a random vector of size 10
	x = A \ b 		# x is the solutions to the linear system Ax=b
	norm(A * x  - b)
end

# ╔═╡ e658ba2e-235c-11eb-05bd-c7467b4c1673
md"""
A few things that are noteworthy: 
- `A` is a `Matrix` type, and `b` is a `Vector` type.
- The transpose function creates a matrix of type `Adjoint`.
- `\` is always the recommended way to solve a linear system. You almost never want to call the `inv` function.
"""

# ╔═╡ 14f62b20-235d-11eb-3a2a-4971e460cbf8
[typeof(A), typeof(b), typeof(rand(1, 10)), typeof(Atranspose)]

# ╔═╡ 3c839880-235d-11eb-229f-297e5e9f36e3
Matrix{Float64} == Array{Float64,2}

# ╔═╡ 5aa3d870-235d-11eb-0f1d-0fd6bcee4fe9
Vector{Float64} == Array{Float64,1}

# ╔═╡ 5e1d6ed0-235d-11eb-1352-e78e7c411e9d
Atranspose

# ╔═╡ 67a13180-235d-11eb-0f38-9d32ca09ec07
md"""
> `adjoint` in julia is a lazy adjoint -- often, we can easily perform Linear Algebra operations such as `A*A'` without actually transposing the matrix.
>
> `adjoint(A)`
>
> Lazy adjoint (conjugate transposition) (also postfix '). Note that adjoint is applied recursively to elements.
>
> This operation is intended for linear algebra usage - for general data manipulation see [permutedims](@ref Base.permutedims).
> ### Examples
>
> ```julia
> julia> A = [3+2im 9+2im; 8+7im  4+6im]
> 2×2 Array{Complex{Int64},2}:
>  3+2im  9+2im
>  8+7im  4+6im
> 
> julia> adjoint(A)
> 2×2 Adjoint{Complex{Int64},Array{Complex{Int64},2}}:
>  3-2im  8-7im
>  9-2im  4-6im
>```
"""

# ╔═╡ a5467640-23c5-11eb-331a-53c5d515e774
Atranspose.parent

# ╔═╡ bb532550-23c5-11eb-11cc-4f2516a1fe44
sizeof(A)

# ╔═╡ c401e4c0-23c5-11eb-0608-6fd36f45142a
md"**NOTE:** ☝️ That's because it's an array of Float64's, each is of size 8 bytes, and there are 10*10 numbers."

# ╔═╡ c97b1bb0-23c5-11eb-303a-0ffd60abc378
# To actually copy the matrix:
A_copy = copy(Atranspose)

# ╔═╡ eeff4b90-23c5-11eb-2fed-ef8b2af19020
md"""
> `\(x, y)`
>
> `adjoint(A)`
>
> Left division operator: multiplication of `y` by the inverse of `x` on the left. Gives floating-point results for integer arguments.
>
> ### Examples
>
> ```julia
> julia> 3 \ 6
> 2.0
>
> julia> inv(3) * 6
> 2.0
>
> julia> A = [4 3; 2 1]; x = [5, 6];
>
> julia> A \ x
> 2-element Array{Float64,1}:
>  6.5
> -7.0
>
> julia> inv(A) * x
> 2-element Array{Float64,1}:
>  6.5
> -7.0
> ```
> Matrix division using a polyalgorithm. For input matrices `A` and `B`, the result `X` is such that `A*X == B` when `A` is square. The solver that is used depends upon the structure of `A`. If `A` is upper or lower triangular (or diagonal), no factorization of `A` is required and the system is solved with either forward or backward substitution. For non-triangular square matrices, an LU factorization is used.
>
> For rectangular `A` the result is the minimum-norm least squares solution computed by a pivoted QR factorization of `A` and a rank estimate of `A` based on the `R` factor.
>
> When `A` is sparse, a similar polyalgorithm is used. For indefinite matrices, the LDLt factorization does not use pivoting during the numerical factorization and therefore the procedure can fail even for invertible matrices.
"""

# ╔═╡ a8ab56b0-23c6-11eb-3b00-c1281be96bed
md"""
### 🟢Factorizations
A common tool used in Linear Algebra is matrix factorizations. These factorizations are often used to solve linear systems like `Ax = b`, and as we will see later in this tutorial... `Ax = b` comes up in a lot of Data Science problems.
"""

# ╔═╡ dbbe4a30-23c6-11eb-0dcc-87309304b9c6
md"""
#### LU factorization
$L*U = P*A$
"""

# ╔═╡ 188d0050-23c7-11eb-271e-bfcc23d636b3
luA = lu(A)

# ╔═╡ 200d2ad0-23c7-11eb-2182-41117fe94a8e
norm(luA.L * luA.U - luA.P * A)

# ╔═╡ 6d046240-23c7-11eb-1db3-398f2e163756
md"""
#### QR factorization
$Q*R = A$
"""

# ╔═╡ 7cac41e0-23c7-11eb-29b6-8b670be90e44
qrA = qr(A)

# ╔═╡ 8dc7aeb0-23c7-11eb-2026-456af59e8e43
norm(qrA.Q * qrA.R - A)

# ╔═╡ 9bf1a0e0-23c7-11eb-1c82-836dcf0996fa
md"""
#### Cholesky factorization
_Note that `A` needs to be symmetric positive definite_

$L*L' = A$
"""

# ╔═╡ c9fb5d50-23c7-11eb-1faf-1bb8387151ee
isposdef(A)

# ╔═╡ cb97811e-23c7-11eb-25d3-c1d154c4144c
cholA = cholesky(A)

# ╔═╡ db98a450-23c7-11eb-3f1e-61d52efbf918
norm(cholA.L * cholA.U - A)

# ╔═╡ 191833e0-23c8-11eb-1187-0fb4eb218f72
cholA.L == cholA.U'

# ╔═╡ 1e384e50-23c8-11eb-05a5-51099b6b762a
factorize(A)

# ╔═╡ 2a742c20-23c8-11eb-2a7c-eba3f3dfaf8d
diagm(1=>[1, 2, 3])

# ╔═╡ 05a0b930-23c9-11eb-3855-0ba2c6e25948
Diagonal([1, 2, 3])

# ╔═╡ 1ad46d60-23c9-11eb-33a8-430ac55a78bc
md"👇 `I` is a function"

# ╔═╡ 2b890210-23c9-11eb-36ba-ed73111fdd9e
I(3)

# ╔═╡ 3a3fa3de-23c9-11eb-0d1a-8b8889daa8c2
md"""
### 🟢Sparse Linear Algebra
Sparse matrices are stored in Compressed Sparse Column (CSC) form
"""

# ╔═╡ 49125930-23c9-11eb-1c68-095d26b3bb2b
S = sprand(5, 5, 2 / 5)

# ╔═╡ 56bf0510-23c9-11eb-08d4-af7b62640d9e
S.rowval

# ╔═╡ 886b63b0-23c9-11eb-0ec3-e3d9fac12f36
Matrix(S)

# ╔═╡ 90fed2f0-23c9-11eb-0d72-2b9b9eef29d9
S.colptr

# ╔═╡ 947d4b50-23c9-11eb-1162-1573fc40bb7f
S.m

# ╔═╡ bcc77900-23c9-11eb-1d94-0f9e028ae605
md"""
### 🟢Images as matrices
Let's get to the more "data science-y" side. We will do so by working with images (which can be viewed as matrices), and we will use the `SVD` decomposition.

First let's load an image. I chose this image as it has a lot of details.
"""

# ╔═╡ c0fa7450-23c9-11eb-3d13-0f2ad01664f3
X1 = load("data/khiam-small.jpg")

# ╔═╡ cd9b2e70-23c9-11eb-1778-379ae8f83d7c
typeof(X1)

# ╔═╡ d92f5a90-23c9-11eb-225d-63d63babb2f8
X1[1, 1] # this is pixel [1, 1]

# ╔═╡ e34af072-23c9-11eb-0a8d-010298bdda8f
md"We can easily convert this image to gray scale."

# ╔═╡ ecb7a8b0-23c9-11eb-1944-4fcbff1f1a57
Xgray = Gray.(X1)

# ╔═╡ f26695a0-23c9-11eb-1dfa-b3be1bd7819e
md"We can easily extract the RGB layers from the image. We will make use of the `reshape` function below to reshape a vector to a matrix."

# ╔═╡ f8c6aa20-23c9-11eb-3f25-4bcb7fb913fd
begin
	R = map(i -> X1[i].r, 1: length(X1))
	R = Float64.(reshape(R, size(X1)...))
	
	G = map(i -> X1[i].g, 1: length(X1))
	G = Float64.(reshape(G, size(X1)...))
	
	B = map(i -> X1[i].b, 1: length(X1))
	B = Float64.(reshape(B, size(X1)...))
end

# ╔═╡ aa045b62-23e4-11eb-25aa-f58cf9efd9bb
begin
	Z = zeros(size(R)...)
	RGB.(Z, G, Z)
end

# ╔═╡ c23e98d0-23e4-11eb-317e-af96e5367d4a
md"We can easily obtain the `Float64` values of the grayscale image."

# ╔═╡ ca79d5ee-23e4-11eb-2907-2b28d80c618f
Xgray_values = Float64.(Xgray)

# ╔═╡ d0d22242-23e4-11eb-302c-4b7849fe9c23
md"Next, we will downsample this image using the SVD. First, let's obtain the SVD decomposition."

# ╔═╡ d97e228e-23e4-11eb-2ff0-bd21f2fa4723
SVD_V = svd(Xgray_values)

# ╔═╡ dd0881d0-23e4-11eb-3b40-c5f7bb5d0a44
norm(SVD_V.U * diagm(SVD_V.S) * SVD_V.V' - Xgray_values)

# ╔═╡ 0f977840-23e5-11eb-0267-83d529a9d193
md"🌌 Use the top 4 singular vectors/values to form a new image:"

# ╔═╡ 4a78dd50-23e5-11eb-1d93-f701748bb839
begin
	i = 1: 100
	u1 = SVD_V.U[:, i]
	v1 = SVD_V.V[:, i]
	img1 = u1 * spdiagm(0=>SVD_V.S[i]) * v1'
end

# ╔═╡ b08c4dc2-23e5-11eb-1af8-87e52517733a
Gray.(img1)

# ╔═╡ 30682040-23e7-11eb-2635-9d7dd497f3ff
md"This looks almost identical to the original image, even though it's not identical to the original image (and we can see that from the norm difference)."

# ╔═╡ ba0cca30-23e7-11eb-1285-1d4f34cb5d81
norm(Xgray_values-img1)

# ╔═╡ 466e188e-23e7-11eb-0d50-ab6d39d99bdb
md"🔍 Our next problem will still be related to images, but this time we will solve a simple form of the face recognition problem. Let's get the data first."

# ╔═╡ d3d26290-23e7-11eb-2386-2964c1f9acde
M = matread("data/face_recog_qr.mat")

# ╔═╡ d9cfbda0-23e7-11eb-02d3-8900a9e39c40
md"Each vector in `M[\"V2\"]` is a fase image. Let's reshape the first one and take a look."

# ╔═╡ e148e340-23e7-11eb-3953-237964d43aa1
begin
	q = reshape(M["V2"][:, 1], 192, 168)
	Gray.(q)
end

# ╔═╡ 1bd7bb80-23e8-11eb-159b-558885048502
md"""
🔉 Now we will go back to the vectorized version of this image, and try to select the images that are most similar to it from the "dictionary" matrix. Let's use `b = q[:]` to be the query image. Note that the notation `[:]` vectorizes a matrix column wise.
"""

# ╔═╡ 38517760-23e8-11eb-3bf8-411aa1075f81
b1 = q[:]

# ╔═╡ 4bae48b2-23e8-11eb-0a46-57d125ee95f9
md"""
We will remove the first image from the dictionary. The goal is to find the solution of the linear system `Ax=b` where `A` is the dictionary of all images. In face recognition problem we really want to minimize the norm differece `norm(Ax-b)` but the `\` actually solves a least squares problem when the matrix at hand is not invertible.
"""

# ╔═╡ 5ba97870-23e8-11eb-2f3f-531aa27a45f0
begin
	A1 = M["V2"][:, 2: end]
	x1 = A1 \ b1 # Ax = b
	Gray.(reshape(A1 * x1, 192, 168))
end

# ╔═╡ 690481e0-23e8-11eb-01bc-29c4f2e368bb
norm(A1 * x1 - b1)

# ╔═╡ a9dd4e90-23e8-11eb-2e77-13f949c7b075
md"This was an easy problem. Let's try to make the picture harder to recover. We will add some random error."

# ╔═╡ b8b31120-23e8-11eb-2258-0592a35cb0cb
begin
	qv = q + rand(size(q, 1), size(q, 2)) * 0.5
	qv = qv ./ maximum(qv)
	Gray.(qv)
end

# ╔═╡ e4eeeb5e-23e8-11eb-2588-111c4421c48d
begin
	bv = qv[:]
	xv = A1 \bv
	Gray.(reshape(A1 * xv, 192, 168))
end

# ╔═╡ 0c39f890-23e9-11eb-21d7-178bf36471d1
norm(A1 * xv - bv)

# ╔═╡ 13e79bb0-23e9-11eb-00c8-41c192576b43
md"_The error is so much bigger this time._"

# ╔═╡ 199bb8c0-23e9-11eb-3554-3d7e7b4aebd8
md"""
# Finally...
After finishing this notebook, you should be able to:
- [ ] reshape and vectorize a matrix
- [ ] apply basic linear algebra operations such as transpose, matrix-matrix product, and solve a linear systerm
- [ ] call a linear algebra factorization on your matrix
- [ ] use SVD to created a compressed version of an image
- [ ] solve the face recognition problem via a least square approach
- [ ] create a sparse matrix, and call the components of the Compressed Sparse Column storage
- [ ] list a few types of matrices Julia uses (diagonal, upper triangular,...)
- [ ] (unrelated to linear algebra): load an image, convert it to grayscale, and extract the RGB layers
"""

# ╔═╡ Cell order:
# ╟─56cfadb2-2357-11eb-3e5c-2dc86275630e
# ╟─cc4a4b40-2357-11eb-2e5e-cdbb8424f76c
# ╠═be373c70-2357-11eb-1e3e-39b91320918a
# ╠═8e43d110-235a-11eb-3415-674c3aeb39b9
# ╟─579794c0-235b-11eb-1298-a7f5b27db5f8
# ╠═6486c790-235c-11eb-2461-d74d43dbf622
# ╠═8ab233f0-235c-11eb-0830-7d3f4d3590f4
# ╠═b9158210-235c-11eb-1dee-43889fe78fd6
# ╟─e658ba2e-235c-11eb-05bd-c7467b4c1673
# ╠═14f62b20-235d-11eb-3a2a-4971e460cbf8
# ╠═3c839880-235d-11eb-229f-297e5e9f36e3
# ╠═5aa3d870-235d-11eb-0f1d-0fd6bcee4fe9
# ╠═5e1d6ed0-235d-11eb-1352-e78e7c411e9d
# ╟─67a13180-235d-11eb-0f38-9d32ca09ec07
# ╠═a5467640-23c5-11eb-331a-53c5d515e774
# ╠═bb532550-23c5-11eb-11cc-4f2516a1fe44
# ╟─c401e4c0-23c5-11eb-0608-6fd36f45142a
# ╠═c97b1bb0-23c5-11eb-303a-0ffd60abc378
# ╟─eeff4b90-23c5-11eb-2fed-ef8b2af19020
# ╟─a8ab56b0-23c6-11eb-3b00-c1281be96bed
# ╟─dbbe4a30-23c6-11eb-0dcc-87309304b9c6
# ╠═188d0050-23c7-11eb-271e-bfcc23d636b3
# ╠═200d2ad0-23c7-11eb-2182-41117fe94a8e
# ╟─6d046240-23c7-11eb-1db3-398f2e163756
# ╠═7cac41e0-23c7-11eb-29b6-8b670be90e44
# ╠═8dc7aeb0-23c7-11eb-2026-456af59e8e43
# ╟─9bf1a0e0-23c7-11eb-1c82-836dcf0996fa
# ╠═c9fb5d50-23c7-11eb-1faf-1bb8387151ee
# ╠═cb97811e-23c7-11eb-25d3-c1d154c4144c
# ╠═db98a450-23c7-11eb-3f1e-61d52efbf918
# ╠═191833e0-23c8-11eb-1187-0fb4eb218f72
# ╠═1e384e50-23c8-11eb-05a5-51099b6b762a
# ╠═2a742c20-23c8-11eb-2a7c-eba3f3dfaf8d
# ╠═05a0b930-23c9-11eb-3855-0ba2c6e25948
# ╟─1ad46d60-23c9-11eb-33a8-430ac55a78bc
# ╠═2b890210-23c9-11eb-36ba-ed73111fdd9e
# ╟─3a3fa3de-23c9-11eb-0d1a-8b8889daa8c2
# ╠═49125930-23c9-11eb-1c68-095d26b3bb2b
# ╠═56bf0510-23c9-11eb-08d4-af7b62640d9e
# ╠═886b63b0-23c9-11eb-0ec3-e3d9fac12f36
# ╠═90fed2f0-23c9-11eb-0d72-2b9b9eef29d9
# ╠═947d4b50-23c9-11eb-1162-1573fc40bb7f
# ╟─bcc77900-23c9-11eb-1d94-0f9e028ae605
# ╠═c0fa7450-23c9-11eb-3d13-0f2ad01664f3
# ╠═cd9b2e70-23c9-11eb-1778-379ae8f83d7c
# ╠═d92f5a90-23c9-11eb-225d-63d63babb2f8
# ╟─e34af072-23c9-11eb-0a8d-010298bdda8f
# ╠═ecb7a8b0-23c9-11eb-1944-4fcbff1f1a57
# ╟─f26695a0-23c9-11eb-1dfa-b3be1bd7819e
# ╠═f8c6aa20-23c9-11eb-3f25-4bcb7fb913fd
# ╠═aa045b62-23e4-11eb-25aa-f58cf9efd9bb
# ╟─c23e98d0-23e4-11eb-317e-af96e5367d4a
# ╠═ca79d5ee-23e4-11eb-2907-2b28d80c618f
# ╟─d0d22242-23e4-11eb-302c-4b7849fe9c23
# ╠═d97e228e-23e4-11eb-2ff0-bd21f2fa4723
# ╠═dd0881d0-23e4-11eb-3b40-c5f7bb5d0a44
# ╟─0f977840-23e5-11eb-0267-83d529a9d193
# ╠═4a78dd50-23e5-11eb-1d93-f701748bb839
# ╠═b08c4dc2-23e5-11eb-1af8-87e52517733a
# ╟─30682040-23e7-11eb-2635-9d7dd497f3ff
# ╠═ba0cca30-23e7-11eb-1285-1d4f34cb5d81
# ╟─466e188e-23e7-11eb-0d50-ab6d39d99bdb
# ╠═d3d26290-23e7-11eb-2386-2964c1f9acde
# ╟─d9cfbda0-23e7-11eb-02d3-8900a9e39c40
# ╠═e148e340-23e7-11eb-3953-237964d43aa1
# ╟─1bd7bb80-23e8-11eb-159b-558885048502
# ╠═38517760-23e8-11eb-3bf8-411aa1075f81
# ╟─4bae48b2-23e8-11eb-0a46-57d125ee95f9
# ╠═5ba97870-23e8-11eb-2f3f-531aa27a45f0
# ╠═690481e0-23e8-11eb-01bc-29c4f2e368bb
# ╟─a9dd4e90-23e8-11eb-2e77-13f949c7b075
# ╠═b8b31120-23e8-11eb-2258-0592a35cb0cb
# ╠═e4eeeb5e-23e8-11eb-2588-111c4421c48d
# ╠═0c39f890-23e9-11eb-21d7-178bf36471d1
# ╟─13e79bb0-23e9-11eb-00c8-41c192576b43
# ╟─199bb8c0-23e9-11eb-3554-3d7e7b4aebd8
