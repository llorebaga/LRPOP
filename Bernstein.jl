include(joinpath(@__DIR__, "LRPOP.jl"))  
import .LRPOP
using Random 

# Fix random seed for reproducibility
rng = MersenneTwister(10);       

# Set parameters
R = 2;
d = 2;
n = 2;
r = ceil(Int, (d+1)/2);
weights = ones(R);
alpha = ones(Float64, n);
alpha_n = fill(1.0/n, n);

# Initialize solver to study running time more accurately
f_coeffs_R = LRPOP.random_rank_components(R, n, d; alpha);
opt_LRPOP_xt, t_LRPOP_xt = LRPOP.solve_LRPOP(R, n, d, r; alpha, weights, f_coeffs_R);

# Set experiment parameters
n = 100;
r = ceil(Int, (d+1)/2);
alpha = ones(Float64, n);

# Bernstein coefficients for f_coeffs_R
# Multiply polynomials in x given as low→high coefficient vectors
function poly_mul(a::Vector{Float64}, b::Vector{Float64})
    na, nb = length(a), length(b)
    c = zeros(Float64, na + nb - 1)
    @inbounds for i in 1:na
        ai = a[i]
        for j in 1:nb
            c[i+j-1] += ai * b[j]
        end
    end
    return c
end

# (a1 * x + a0)^k as low→high
function poly_pow_affine(a1::Float64, a0::Float64, k::Int)
    k == 0 && return [1.0]
    base = [a0, a1]  # a0 + a1 x
    res = [1.0]
    for _ in 1:k
        res = poly_mul(res, base)
    end
    return res
end

# Scaled Bernstein basis:
# Return as (d+1)-vector, high→low, to match convention.
function scaled_bernstein_coeffs(j::Int, d::Int, n::Int)
    s = sqrt(float(n))

    # B_{j,d}(y) with y = (s x + 1)/2:
    # = binom(d,j)/2^d * (s x + 1)^j * (1 - s x)^(d-j)
    c1 = poly_pow_affine(s, 1.0, j)       # (s x + 1)^j
    c2 = poly_pow_affine(-s, 1.0, d-j)    # (1 - s x)^(d-j)
    c  = poly_mul(c1, c2)                 # low→high, degree d

    factor = binomial(d, j) / (2.0^d)
    @inbounds for k in 1:length(c)
        c[k] *= factor
    end

    # Convert to high→low for LRPOP_full
    return reverse(c)
end

# Build rank-r polynomial coefficients f_coeffs_R for Bernstein case
function make_bernstein_case_coeffs(R::Int, n::Int, d::Int;
                                    delta::Real = 1.0,
                                    rng::AbstractRNG = Random.GLOBAL_RNG,
                                    weights::AbstractVector{<:Real} = ones(R))

    # Precompute scaled Bernstein basis polys for this (d, n)
    bernstein_scaled = [scaled_bernstein_coeffs(j, d, n) for j in 0:d]

    # f_coeffs_R: R × n array of (d+1)-vectors (high→low)
    f_coeffs_R = Vector{Vector{Vector{Float64}}}(undef, R)

    for l in 1:R
        comps = Vector{Vector{Float64}}(undef, n)

        for i in 1:n
            # Bernstein coefficients b_{l,i,j}
            # b_0 = 1, others in [1 + δ/n, 1 + 2δ/n]
            b = zeros(Float64, d+1)
            b[1] = 1.0
            for j in 1:d
                b[j+1] = 1.0 + (delta / n) + (delta / n) * rand(rng)
            end

            # p_{l,i}(x) = sum_j b_j * B_{j,d}((√n x + 1)/2)
            # combine precomputed basis polynomials (high→low)
            cs = zeros(Float64, d+1)
            @inbounds for j in 0:d
                cs .+= b[j+1] .* bernstein_scaled[j+1]
            end

            comps[i] = cs
        end

        f_coeffs_R[l] = comps
    end

    # Analytic minimizer: x_i = -1/√n
    x_star = fill(-1.0 / sqrt(float(n)), n)

    # At x_star, each p_{l,i} = b_0 = 1 ⇒ each product = 1
    # ⇒ p(x_star) = sum_l w[l]
    p_star = sum(float.(weights))

    return f_coeffs_R, x_star, p_star
end

# Bernstein basis for box of size 1
function scaled_bernstein_coeffs_radius(j::Int, d::Int, Rbox::Real)
    @assert 0 <= j <= d
    s = 1.0 / float(Rbox)   # because y = (s*x + 1)/2

    c1 = poly_pow_affine(s, 1.0, j)        # (s x + 1)^j
    c2 = poly_pow_affine(-s, 1.0, d-j)     # (1 - s x)^(d-j)
    c  = poly_mul(c1, c2)                  # low→high

    factor = binomial(d, j) / (2.0^d)
    @inbounds for k in 1:length(c)
        c[k] *= factor
    end
    return reverse(c)  # high→low
end

# Convenience wrapper for the new box x^2 <= 1 (Rbox=1):
scaled_bernstein_coeffs_box1(j::Int, d::Int) = scaled_bernstein_coeffs_radius(j, d, 1.0)

function make_bernstein_coeffs_box1(R::Int, n::Int, d::Int;
                                         delta::Real = 1.0,
                                         rng::AbstractRNG = MersenneTwister(10),
                                         weights::AbstractVector{<:Real} = ones(R))

    # Basis for x ∈ [-1,1]
    bernstein_scaled = [scaled_bernstein_coeffs_box1(j, d) for j in 0:d]

    f_coeffs_R = Vector{Vector{Vector{Float64}}}(undef, R)
    for l in 1:R
        comps = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            b = zeros(Float64, d+1)
            b[1] = 1.0
            for j in 1:d
                # keep gaps to avoid blowup in products
                b[j+1] = 1.0 + (delta / n) + (delta / n) * rand(rng)
            end

            cs = zeros(Float64, d+1)
            @inbounds for j in 0:d
                cs .+= b[j+1] .* bernstein_scaled[j+1]
            end
            comps[i] = cs
        end
        f_coeffs_R[l] = comps
    end

    # New analytic minimizer on [-1,1]^n:
    x_star = fill(-1.0, n)

    p_star = sum(float.(weights))
    return f_coeffs_R, x_star, p_star
end


# Find minimum from random sampling
function random_min_sample(
    f_coeffs_R::Vector{Vector{Vector{Float64}}},
    weights::AbstractVector,
    n::Int;
    nsamples::Int = 100,
    rng::AbstractRNG = MersenneTwister(10),
)

    best_val = +Inf
    best_x   = zeros(Float64, n)

    for k in 1:nsamples
        # x ∈ [-1, 1]^n
        x = 2 .* rand(rng, n) .- 1

        # Evaluate p(x) = sum_l w[l] * ∏_i f_{l,i}(x[i])
        val = LRPOP.eval_rankR_poly(f_coeffs_R, weights, x)

        if val < best_val
            best_val = val
            best_x .= x
        end
    end 

    return best_val, best_x
end

# Find minimum from gradient descent
# Evaluate univariate poly and its derivative (coeffs are high→low: c_d,...,c_0)
function eval_poly_and_deriv(cs::AbstractVector{<:Real}, x::Real)
    val = cs[1]
    der = 0.0
    @inbounds for k in 2:length(cs)
        der = der * x + val      # derivative Horner
        val = val * x + cs[k]    # value Horner
    end
    return val, der
end

# Evaluate rank-R polynomial and its gradient at x
function eval_grad_rankR!(
    grad::AbstractVector{Float64},
    f_coeffs_R::Vector{Vector{Vector{Float64}}},
    weights::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
)
    R = length(f_coeffs_R)
    n = length(x)
    @assert length(grad) == n
    @assert length(weights) == R

    fill!(grad, 0.0)
    total_val = 0.0

    # temporary buffers per rank
    fi   = Vector{Float64}(undef, n)   # p_{l,i}(x_i)
    fip  = Vector{Float64}(undef, n)   # p'_{l,i}(x_i)
    pref = Vector{Float64}(undef, n)   # prefix products of fi
    suff = Vector{Float64}(undef, n)   # suffix products of fi

    @inbounds for l in 1:R
        # evaluate all univariate factors and their derivatives
        for i in 1:n
            cs = f_coeffs_R[l][i]
            fi[i], fip[i] = eval_poly_and_deriv(cs, x[i])
        end

        # prefix and suffix products of fi
        pref[1] = fi[1]
        for i in 2:n
            pref[i] = pref[i-1] * fi[i]
        end
        suff[n] = fi[n]
        for i in (n-1):-1:1
            suff[i] = suff[i+1] * fi[i]
        end

        prod_all = pref[n]
        w = float(weights[l])
        total_val += w * prod_all

        # gradient contributions
        for i in 1:n
            # product over all except i itself
            if i == 1
                prod_except_i = suff[2]
            elseif i == n
                prod_except_i = pref[n-1]
            else
                prod_except_i = pref[i-1] * suff[i+1]
            end

            # All products are evaluated at one point and for i we have the derivative (fip[i])
            grad[i] += w * prod_except_i * fip[i] # we add the grad contribution from rank l
        end
    end

    return total_val
end

# Projected gradient descent on box [-1,1]^n
function projected_gd(
    f_coeffs_R::Vector{Vector{Vector{Float64}}},
    weights::AbstractVector{<:Real},
    n::Int;
    nsweeps::Int = 2000,
    η::Float64 = 1e-2,
    nrestarts::Int = 5,
    rng::AbstractRNG = MersenneTwister(123),
)

    t0 = time();
    x = zeros(Float64, n)
    g = zeros(Float64, n)
    best_val = +Inf
    best_x   = zeros(Float64, n)

    for rstart in 1:nrestarts
        # random initial point in [-1,1]^n
        @inbounds for i in 1:n
            x[i] = 2 * rand(rng) - 1
        end

        val = eval_grad_rankR!(g, f_coeffs_R, weights, x)

        for k in 1:nsweeps
            # gradient step + projection onto [-1,1]^n
            @inbounds for i in 1:n
                x[i] -= η * g[i]
                x[i] = max(-1.0, min(1.0, x[i]))
            end
            val = eval_grad_rankR!(g, f_coeffs_R, weights, x)
        end

        if val < best_val
            best_val = val
            best_x .= x
        end
    end

    t1 = time() - t0;

    return best_val, best_x, t1
end


# Get random coefficients in Bernstein basis on box [-1,1]^n
f_coeffs_R, x_star, p_star = make_bernstein_coeffs_box1(R, n, d; delta=1, rng=rng, weights=weights)

# Solve
opt_LRPOP, t_LRPOP, psd_diag, _ = LRPOP.solve_LRPOP(R, n, d, r; alpha, weights, f_coeffs_R);
random_min_val, _ = random_min_sample(f_coeffs_R, weights, n; nsamples=10^5, rng=rng);
gd_val, gd_x, t_gd = projected_gd(f_coeffs_R, weights, n;
                            nsweeps=1000, η=1e-2, nrestarts=1000, rng=rng)


println("Optimal value xt: ", opt_LRPOP," Random sampling: ", random_min_val, " Analytic optimum: ", p_star, "   Time xt: ", t_LRPOP," Smallest eigenvalue: ", psd_diag);
println("Gradient descent value: ", gd_val, " Time: ", t_gd);
