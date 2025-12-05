module LRPOP

using Graphs
using CliqueTrees
using TreeWidthSolver
using AbstractTrees

using JuMP
using LinearAlgebra
using MosekTools      
using DynamicPolynomials
using Random
using Dates
using TSSOS
using MathOptInterface
using Hypatia

const OPT = MosekTools.Optimizer
const MOI = MathOptInterface

high_precision_opt = JuMP.optimizer_with_attributes(
    MosekTools.Optimizer,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS"   => 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS"   => 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED"  => 1e-16,
)


export  create_graph, clique_decomposition, variables_graph_xt, node_to_var,
        ztuple, addexp, exp_set, add_moment_vars, moment_matrix_affine, 
        build_psd_per_clique, embed_and_get, tie_overlap_two, tie_all_overlaps, parent_from_tree, tie_overlaps_on_tree,
        find_clique_with, exp_from_mon, totaldeg, poly_total_degree, add_all_moment_eqs, 
            add_t1_eq_fx, add_t1_eq_fx_all, add_t_eq_tprev_times_fx, add_t_eq_tprev_times_fx_all,
        add_localizing_psd, find_smallest_clique_with, add_ball_localizers_on_x,
        supnorm_poly_on_box_hilo, random_univariate_coeffs_balanced, random_rank_components,
        build_full_poly, diagnose_psd_blocks, solve_LRPOP, solve_TSSOS, eval_unipoly, eval_rankR_poly


### Graph and decomposition functions

## Create graph G_{r,n}
function create_graph(r,n)
    G = SimpleGraph((r+1)*n);
    for i in 1:n, j in 1:r
            add_edge!(G,i,j*n+i);
    end
    for i in 1:(n-1), j in 1:r
            add_edge!(G,j*n+i,j*n+i+1);
            add_edge!(G,j*n+i,i+1);
    end
    return G
end

## Find clique decomposition. Returns cliques, separators, vertex labels and the tree
function clique_decomposition(G)
    A = Matrix{Int}(adjacency_matrix(G))
    alg = SafeRules(BT(), MMW(), MF())
    label, T = cliquetree(A; alg)   # label: permuted vertices, T: clique tree
    # The cliquetree function permutes the vertices, assigning different numbers
    
    invlabel = invperm(label)       # Find permuted vertices from originals
    map_orig(vs) = sort(label[vs])  # Recover original vertices from permuted

    # Cliques and separations
    cliques = [map_orig(Cperm) for Cperm in T ]  # Cperm are the cliques with permuted indices
    seps_to_parent = Vector{Vector{Int}}(undef, length(T))
    for i in 1:length(T)
        Sepperm = (i == 1) ? Int[] : separator(T, i)  # Separators to parent with permuted indices
        seps_to_parent[i] = map_orig(Sepperm)         # Separators with original indices
    end
    return cliques, seps_to_parent, label, T
end

## Assign variables (x_i and t_{l,i}) to the graph vertices
function variables_graph_xt(r, n)
    x = [ Symbol("x_", i) for i in 1:n ]
    t = [ [ Symbol("t_", l, "_", i) for i in 1:n ] for l in 1:r ]

    N = (r+1)*n
    ver_to_var = Vector(undef, N)

    # Create the vector that goes from vertex number to variable
    # This follows from the definition of the vertices in create_graph(r,n)
    for i in 1:n
        ver_to_var[i] = x[i]
    end
    for l in 1:r, i in 1:n
        ver_to_var[l*n + i] = t[l][i]
    end
    
    return ver_to_var, x, t
end

## Assign bag of variables to cliques and separators
function node_to_var(cliques, seps_to_parent, ver_to_var)
    # Do it for all vectors in cliques and separators
    clique_vars = [Symbol[ver_to_var[n] for n in cl] for cl in cliques];
    sep_vars = [Symbol[ver_to_var[n] for n in sep] for sep in seps_to_parent];

    return clique_vars, sep_vars
end


### Create moment matrices

## Create the M[1,1] element 1, as a tuple of 0s, (the exponents)
ztuple(m::Int) = ntuple(_ -> 0, m)
addexp(a::NTuple{M,Int}, b::NTuple{M,Int}) where {M} = ntuple(i->a[i]+b[i], M)

## Create all exponent tuples for m variables, and up to degree deg
function exp_set(m::Int, deg::Int)
    res = NTuple{m,Int}[] # Holds tuples of length m, the exponents
    e = zeros(Int, m) # Tuple to fill with exponents
    function rec(pos::Int, left::Int) # Recursion to fill e
        # pos is the coordinate of the exponent and left the total degree remaining
        if pos == m # last coordinate
            e[pos] = left # give it all degree to last position
            push!(res, Tuple(e))
            return
        end
        for v in 0:left # Iterate over all possible degrees for the corresponding exponent
            e[pos] = v # Give the appropiate exponent
            rec(pos+1, left-v) # Move to next position
        end
    end
    for t in 0:deg # Create all the monomials of degree up to deg
        fill!(e, 0) # Restart e to a clean vector
        rec(1, t) # Start the recursion at the first exponent
    end
    return res
end

## Create JuMP variables y_e up to e<= 2r for the variables listed in vars
function add_moment_vars!(model::Model, vars::Vector{Symbol}, deg2::Int) 
    m = length(vars)
    E = exp_set(m, deg2) # Gives all monomial exponents in moment matrix
    y = Dict{NTuple{m,Int}, JuMP.VariableRef}() # Moment variables
    for e in E
        # We name them e.g. y_x_2^3*x_4
        nm = join([string(vars[i], "^", e[i]) for i in 1:m if e[i] > 0], "*")
        nm = isempty(nm) ? "1" : nm # For constant y_1
        y[e] = @variable(model, base_name = "y_" * nm)
    end
    return y
end

## Build the moment matrix M_r(y) for monomials up to degree r
function moment_matrix_affine(
    y::Dict{NTuple{m,Int},JuMP.VariableRef}, r::Int
    ) where {m} # To let Julia know we will use m from the definition of y
    B = exp_set(m, r) # Basis exponents
    nb = length(B)
    M = [zero(AffExpr) for _ in 1:nb, _ in 1:nb] # Initialize the moment matrix with 0 to later assign moments
    for i in 1:nb, j in i:nb
        e = ntuple(k -> B[i][k] + B[j][k], m)  # Assign exponent vectors
        # Assign moments to exponent vectors
        M[i,j] = y[e]
        M[j,i] = y[e] # For symmetry
    end
    return M, B
end


### Impose PSD conditions and overlap constraints

## Impose all PSD constraints in every clique of the tree
function build_psd_per_clique!(
    clique_vars::Vector{Vector{Symbol}};
    r::Int, optimizer = OPT
)
    model = Model(optimizer);
    set_silent(model);
    K = length(clique_vars)

    bag_y = Vector{Any}(undef, K)
    bag_M = Vector{Any}(undef, K)
    bag_B = Vector{Any}(undef, K)

    for k in 1:K
        vars = clique_vars[k]
        y = add_moment_vars!(model, vars, 2*r) # all moments up to 2r
        M, B = moment_matrix_affine(y, r) # the moment matrix and its basis

        @constraint(model, Symmetric(M) in PSDCone()) # Impose PSD
        @constraint(model, y[ztuple(length(vars))] == 1) # y_1 = 1

        bag_y[k] = y
        bag_M[k] = M
        bag_B[k] = B
    end
    # Return model, bag_y[k][e] is y_e for clique k, moment matrices and basis per clique
    return model, bag_y, bag_M, bag_B 
end

## Get the moments of the shared variables
function embed_and_get(
    y::Dict{NTuple{m,Int},JuMP.VariableRef},
    clique_vars::Vector{Symbol},
    shared_vars::Vector{Symbol},
    e_shared::NTuple{K,Int}
) where {m,K}
    # Map from variable symbol to position in this clique
    pos = Dict{Symbol,Int}(clique_vars[i] => i for i in 1:length(clique_vars))

    # Start with all-zero exponents in the clique's own variable order
    e_full = zeros(Int, length(clique_vars))

    # Put the shared exponents into the right positions (according to this clique), @inbounds to make sure index exists
    @inbounds for (k, sv) in enumerate(shared_vars)
        e_full[pos[sv]] = e_shared[k]
    end

    # Look up that full exponent tuple in this clique's moment dictionary
    return y[Tuple(e_full)]
end

## Build a position dictionary for fast lookup
_make_pos(vs::Vector{Symbol}) = Dict{Symbol,Int}(vs[i] => i for i in 1:length(vs))

## Helper to embed with precomputed positions
function _embed_with_pos(
    y::Dict{NTuple{m,Int},JuMP.VariableRef},
    pos::Dict{Symbol,Int},
    bag_len::Int,
    shared_vars::Vector{Symbol},
    e_shared::NTuple{K,Int}
) where {m,K}
    e_full = zeros(Int, bag_len)
    @inbounds for (k, sv) in enumerate(shared_vars)
        e_full[pos[sv]] = e_shared[k]
    end
    return y[Tuple(e_full)]
end

## Impose overlap constraints in a pair of cliques sharing variables
function tie_overlap_two!(
    model::JuMP.Model,
    yA::Dict{NTuple{mA,Int},JuMP.VariableRef}, varsA::Vector{Symbol},
    yB::Dict{NTuple{mB,Int},JuMP.VariableRef}, varsB::Vector{Symbol},
    r::Int
) where {mA,mB}
    
    setB = Set(varsB) # Build a set for easy membership check in Julia
    shared = [ z for z in varsA if z in setB ]

    # If nothing in common, nothing to tie
    isempty(shared) && return nothing

    # Enumerate all exponent tuples on the shared variables with e <= 2r
    for eS in exp_set(length(shared), 2*r)
        # Set the equality for the moment defined by shared exponents eS
        @constraint(model,
            embed_and_get(yA, varsA, shared, eS) ==
            embed_and_get(yB, varsB, shared, eS)
        )
    end
    return nothing
end

## Impose overlap condition on all the cliques in a pairwise manner
function tie_all_overlaps!(
    model::JuMP.Model,
    bag_y::Vector{<:Any},
    bag_vars::Vector{Vector{Symbol}},
    r::Int
)
    K = length(bag_vars) # Take the K cliques
    for a in 1:K-1, b in a+1:K # Iterate over all cliques
        # Only tie if they share something
        if !isempty(intersect(bag_vars[a], bag_vars[b]))
            tie_overlap_two!(model, bag_y[a], bag_vars[a],
                                   bag_y[b], bag_vars[b], r)
        end
    end
    return nothing
end

## Build parent indices from the actual clique tree (root has parent 0)
parent_from_tree(T) = begin
    p = fill(0, length(T))
    for j in 1:length(T)
        for c in AbstractTrees.childindices(T, j)
            p[c] = j
        end
    end
    p
end

## Impose overlap equalities following the tree structure
function tie_overlaps_on_tree!(
    model::JuMP.Model,
    bag_y::Vector{<:Any},
    bag_vars::Vector{Vector{Symbol}},
    bag_pos::Vector{Dict{Symbol,Int}},
    sep_vars::Vector{Vector{Symbol}},
    parent::Vector{Int},
    r::Int
)
    K = length(bag_vars)
    for i in 2:K
        j = parent[i]
        shared = sep_vars[i]
        isempty(shared) && continue
        for eS in exp_set(length(shared), 2*r)
            @constraint(model,
                _embed_with_pos(bag_y[j], bag_pos[j], length(bag_vars[j]), shared, eS) ==
                _embed_with_pos(bag_y[i], bag_pos[i], length(bag_vars[i]), shared, eS)
            )
        end
    end
    nothing
end


### Set the lifting equalities

## Finds the first clique that contain the needed variables
function find_clique_with(bag_vars::Vector{Vector{Symbol}},
                          needed::Vector{Symbol})
    need = Set(needed)
    for k in eachindex(bag_vars)
        if all(z -> z in bag_vars[k], need) # Check all needed variables are in the clique k
            return k
        end
    end
end

## Return every clique index whose bag contains all 'needed' variables
function find_all_cliques_with(bag_vars::Vector{Vector{Symbol}},
                               needed::Vector{Symbol})
    need = Set(needed)
    Ks = Int[]
    for k in eachindex(bag_vars)
        if all(z -> z in bag_vars[k], need)
            push!(Ks, k)
        end
    end
    return Ks
end

## Build an exponent tuple for a monomial
function exp_from_mon(vars::Vector{Symbol}, mon::AbstractDict{Symbol,<:Integer})
    m = length(vars)
    pos = Dict{Symbol,Int}(vars[i] => i for i in 1:m)
    e = zeros(Int, m)
    for (v, p) in mon # for variable symbol v and power p
        @assert haskey(pos, v) # Make sure the variable is in the list of variables
        e[pos[v]] = Int(p) # Insert the exponent for variable v
    end
    return Tuple(e) # Return the tuple of exponents
end

## Function to find total degree of monomial and of polynomial
totaldeg(e::NTuple{M,Int}) where {M} = sum(e)
poly_total_degree(terms::Vector{Tuple{Float64,NTuple{M,Int}}}) where {M} =
    maximum(totaldeg(e) for (_, e) in terms)

## Impose E[m(z)*h(z)] = 0 for every monomial m with deg(m) <= 2r - deg(h)
function add_all_moment_eqs!(
    model::Model,
    y::Dict{NTuple{m,Int},JuMP.VariableRef},
    vars::Vector{Symbol},
    h_terms::Vector{Tuple{Float64,NTuple{m,Int}}},
    r::Int
) where {m}
    deg_h = poly_total_degree(h_terms)
    max_deg_m = 2*r - deg_h
    @assert max_deg_m ≥ 0 # Make sure the degree of the monomials is nonnegative

    # enumerate monomials m on this clique up to maximum degree max_deg_m
    for mexp in exp_set(length(vars), max_deg_m)
        lhs = zero(AffExpr) # Initialize equation for moments, left hand side
        for (c, e) in h_terms
            # Go from exponents to moments, to define the equation
            lhs += c * y[ntuple(i -> mexp[i] + e[i], length(vars))]
        end
        @constraint(model, lhs == 0)
    end
    return nothing
end

## Impose the equalities t1 = f1(x1)
function add_t1_eq_fx!(
    model::Model, bag_y::Vector{<:Any}, bag_vars::Vector{Vector{Symbol}},
    r::Int, t1::Symbol, x1::Symbol, coeffs::Vector{<:Real}
)
    k = find_clique_with(bag_vars, [x1, t1]) # Find first clique with both variables
    vars = bag_vars[k]; y = bag_y[k]

    d  = length(coeffs) - 1
    terms = Vector{Tuple{Float64,NTuple{length(vars),Int}}}()
    push!(terms, ( +1.0, exp_from_mon(vars, Dict(t1=>1)) ))
    for (j,c) in enumerate(coeffs)  # Note from high to low the coefficients
        pow = d - (j-1)
        push!(terms, (-float(c),
                      pow>0 ? exp_from_mon(vars, Dict(x1=>pow))
                            : exp_from_mon(vars, Dict{Symbol,Int}()))) # Make the moments from exps
    end
    add_all_moment_eqs!(model, y, vars, terms, r) # Impose the constraint in the model
    return k
end

## For all cliques containing t1 and x1
function add_t1_eq_fx_all!(
    model::Model, bag_y::Vector{<:Any}, bag_vars::Vector{Vector{Symbol}},
    r::Int, t1::Symbol, x1::Symbol, coeffs::Vector{<:Real}
)
    Klist = find_all_cliques_with(bag_vars, [x1, t1])
    for k in Klist
        vars = bag_vars[k]; y = bag_y[k]
        d  = length(coeffs) - 1
        terms = Vector{Tuple{Float64,NTuple{length(vars),Int}}}()
        push!(terms, ( +1.0, exp_from_mon(vars, Dict(t1=>1)) ))
        for (j,c) in enumerate(coeffs)
            pow = d - (j-1)
            push!(terms, pow>0 ? (-float(c), exp_from_mon(vars, Dict(x1=>pow)))
                               : (-float(c), exp_from_mon(vars, Dict{Symbol,Int}())) )
        end
        add_all_moment_eqs!(model, y, vars, terms, r)
    end
    return nothing
end

## General case for t and x
function add_t_eq_tprev_times_fx!(
    model::Model, bag_y::Vector{<:Any}, bag_vars::Vector{Vector{Symbol}},
    r::Int, tprev::Symbol, xi::Symbol, tcurr::Symbol,
    coeffs::Vector{<:Real}
)
    k = find_clique_with(bag_vars, [tprev, xi, tcurr])
    vars = bag_vars[k]; y = bag_y[k]

    d  = length(coeffs) - 1
    terms = Vector{Tuple{Float64,NTuple{length(vars),Int}}}()
    push!(terms, ( +1.0, exp_from_mon(vars, Dict(tcurr=>1)) ))
    for (j,c) in enumerate(coeffs)
        pow = d - (j-1)
        mon = pow>0 ? Dict(tprev=>1, xi=>pow) : Dict(tprev=>1)
        push!(terms, (-float(c), exp_from_mon(vars, mon)))
    end
    add_all_moment_eqs!(model, y, vars, terms, r)
    return k
end

## For all cliques containing tprev, xi, tcurr
function add_t_eq_tprev_times_fx_all!(
    model::Model, bag_y::Vector{<:Any}, bag_vars::Vector{Vector{Symbol}},
    r::Int, tprev::Symbol, xi::Symbol, tcurr::Symbol,
    coeffs::Vector{<:Real}
)
    Klist = find_all_cliques_with(bag_vars, [tprev, xi, tcurr])
    for k in Klist
        vars = bag_vars[k]; y = bag_y[k]
        d  = length(coeffs) - 1
        terms = Vector{Tuple{Float64,NTuple{length(vars),Int}}}()
        push!(terms, ( +1.0, exp_from_mon(vars, Dict(tcurr=>1)) ))
        for (j,c) in enumerate(coeffs)
            pow = d - (j-1)
            mon = pow>0 ? Dict(tprev=>1, xi=>pow) : Dict(tprev=>1)
            push!(terms, (-float(c), exp_from_mon(vars, mon)))
        end
        add_all_moment_eqs!(model, y, vars, terms, r)
    end
    return nothing
end


### Add localizing matrices, e.g. for ball constraints

## Create localizing moment matrix for some inequality g
function add_localizing_psd!(
    model::Model,
    y::Dict{NTuple{m,Int},JuMP.VariableRef},
    vars::Vector{Symbol},
    g_terms::Vector{Tuple{Float64,NTuple{m,Int}}},
    r::Int
) where {m}
    dg = poly_total_degree(g_terms)
    t = r - ceil(Int, dg/2) # Maximum order of the monomials for a localizing matrix of degree 2r

    B = exp_set(length(vars), t)
    nb = length(B)
    L = Array{AffExpr}(undef, nb, nb)
    for i in 1:nb, j in i:nb # Form each entry of the moment matrix from g and monomials
        s = zero(AffExpr)
        a = B[i]; b = B[j]
        for (c,e) in g_terms 
            s += c * y[addexp(e, addexp(a, b))]
        end

        # Moment matrix is symmetric
        L[i,j] = s; L[j,i] = s
    end
    @constraint(model, Symmetric(L) in PSDCone())
    return nothing
end

## Get the index of the smallest clique containing the variables we need
function find_smallest_clique_with(bag_vars::Vector{Vector{Symbol}},
                                   needed::Vector{Symbol})
    need = Set(needed)
    best_k  = nothing
    best_sz = typemax(Int)
    for k in eachindex(bag_vars)
        vs = bag_vars[k]
        if all(z -> z in vs, need)
            if length(vs) < best_sz
                best_k, best_sz = k, length(vs)
            end
        end
    end
    best_k === nothing && error("No clique contains all of: $(collect(need))")
    return best_k
end

# Add localizing constraint on x_i in the smallest clique possible
function add_ball_localizers_on_x!(
    model::Model,
    bag_y::Vector{<:Any},
    bag_vars::Vector{Vector{Symbol}},
    r::Int,
    x_syms::Vector{Symbol},
    R::AbstractVector{<:Real}
)
    @assert length(x_syms) == length(R)
     # Check we have as many radii as variables to bound

    # Iterate over all variables to bound
    for i in eachindex(x_syms)
        xi = x_syms[i]
        k  = find_smallest_clique_with(bag_vars, [xi])
        vars = bag_vars[k]; y = bag_y[k]

        # Set the inequality R_i^2 - x_i^2 ≥ 0
        g = Tuple{Float64,NTuple{length(vars),Int}}[
            ( R[i]^2, exp_from_mon(vars, Dict{Symbol,Int}())),  # in the x^0 monomial exponent
            (-1.0,   exp_from_mon(vars, Dict(xi=>2)))           
        ]
        add_localizing_psd!(model, y, vars, g, r)
    end
    return nothing
end


### Generate random rank-r polynomials with controlled sup-norm on a box

## Create a bound on the polynomial so they do not make the problem ill-conditioned
function supnorm_poly_on_box_hilo(cs::Vector{Float64}, ρ::Float64)
    d = length(cs) - 1
    s = 0.0
    @inbounds for (k, c) in enumerate(cs)  
        pow = d - (k - 1)                   
        s += abs(c) * ρ^pow
    end
    return max(s, 1e-12)
end

## Find the coefficients of n univariate polynomials of degree d
function random_univariate_coeffs_balanced(n::Int, d::Int;
    alpha::Vector{Float64},
    decay::Float64 = 0.7,
    target_sup::Float64 = 1.0,
    rng = MersenneTwister(10)
)
    @assert length(alpha) == n
    ρ = sqrt.(max.(alpha, 0.0)) # radii for x bound, set negatives to 0
    coeffs = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        # sample coefficients (d+1 many)
        lohi = [randn(rng) * decay^k for k in 0:d]
        cs_hilo = reverse(lohi) # We want an order highest to lowest for consistency
        s = supnorm_poly_on_box_hilo(cs_hilo, ρ[i])
        coeffs[i] = (target_sup / s) .* cs_hilo # Normalize the coefficients with the bound
    end
    return coeffs
end

## Find the coefficients for a rank r polynomial
function random_rank_components(R::Int, n::Int, d::Int;
    alpha::Vector{Float64},
    decay::Float64 = 0.7,
    target_sup::Float64 = 1.0,
    rng = MersenneTwister(10)
)
    f_coeffs_R = Vector{Vector{Vector{Float64}}}(undef, R)
    for l in 1:R
        f_coeffs_R[l] = random_univariate_coeffs_balanced(n, d;
            alpha=alpha, decay=decay, target_sup=target_sup, rng=rng)
    end
    return f_coeffs_R
end

# Build the full polynomial 
function build_full_poly(f_coeffs_R::Vector{Vector{Vector{Float64}}}, weights::Vector{Float64})
    R = length(f_coeffs_R); n = length(f_coeffs_R[1])
    @polyvar x[1:n]
    p = zero(x[1])
    for l in 1:R
        term = one(x[1])
        for i in 1:n
            cs = f_coeffs_R[l][i]
            d  = length(cs) - 1
            fi = zero(x[1])
            @inbounds for (k,c) in enumerate(cs)         
                pow = d - (k-1)
                fi += c * x[i]^pow
            end
            term *= fi
        end
        p += weights[l] * term
    end
    return p, x
end


### Check the PSD blocks after the optimization to check if they are really PSD
function diagnose_psd_blocks(
    bag_M::Vector
)
    min_val  = +Inf
    min_bag  = 0
    min_size = 0
    for k in eachindex(bag_M)
        Mk = JuMP.value.(bag_M[k])
        isempty(Mk) && continue
        Mk = 0.5*(Mk + Mk')
        λmin = eigmin(Symmetric(Mk))
        if λmin < min_val
            min_val  = λmin
            min_bag  = k
            min_size = size(Mk, 1)
        end
    end

    return min_val
end


### Solve rank-r problem via LRPOP
function solve_LRPOP(R::Int, n::Int, d::Int,
    r::Int;
    alpha::Vector{Float64},
    weights::Union{Nothing,Vector{Float64}} = nothing,
    f_coeffs_R::Union{Nothing,Vector{Vector{Vector{Float64}}}} = nothing,
    decay::Float64 = 0.7,
    target_sup::Float64 = 1.0,
    seed::Int = 10,
    optimizer = high_precision_opt, 
    psd_check::Bool=true
)
    @assert r ≥ ceil(Int, (d+1)/2) # Now some bags need a +1 in the degree

    t0 = time()

    # Create the graph and decomposition
    G = create_graph(R, n)
    cliques, seps_to_parent, label, T = clique_decomposition(G)

    # Get the variables from the node numbers
    ver_to_var, x, t = variables_graph_xt(R, n)
    clique_vars, sep_vars = node_to_var(cliques, seps_to_parent, ver_to_var)

    # Impose PSD constraints in the clique bags
    model, bag_y, bag_M, bag_B = build_psd_per_clique!(clique_vars; r, optimizer=optimizer)

    # Impose overlap equalities in the bags
    tie_all_overlaps!(model, bag_y, clique_vars, r) 

    # Impose the lifting equalities
    for l in 1:R
        add_t1_eq_fx!(model, bag_y, clique_vars, r, t[l][1], x[1], f_coeffs_R[l][1])
        for i in 2:n
            add_t_eq_tprev_times_fx!(model, bag_y, clique_vars, r,
                                     t[l][i-1], x[i], t[l][i], f_coeffs_R[l][i])
        end
    end

    # Add the localizing moment matrices
    add_ball_localizers_on_x!(model, bag_y, clique_vars, r, x, sqrt.(alpha))

    # Build the objective to optimize
    obj = zero(AffExpr)
    for l in 1:R
        kl = find_clique_with(clique_vars, [t[l][n]])
        varsl = clique_vars[kl]; yl = bag_y[kl]
        e_tln = exp_from_mon(varsl, Dict(t[l][n]=>1))
        obj += weights[l] * yl[e_tln]
    end
    @objective(model, Min, obj)

    set_silent(model)
    optimize!(model)

    t1 = time() - t0;

    psd_diag = nothing
    if psd_check
        psd_diag = diagnose_psd_blocks(bag_M)
    end

    return objective_value(model), t1, psd_diag,  termination_status(model)
end


### Solve problem via TSSOS
function solve_TSSOS(f_coeffs_R::Vector{Vector{Vector{Float64}}}, 
    weights::Vector{Float64}, n::Int, d::Int, alpha::Vector{Float64} = fill(1.0/n, n)
    )
    t0 = time()

    settings = mosek_para()
    settings.time_limit = 200.0 # limit of running time

    p, x = build_full_poly(f_coeffs_R, weights);
    cons = [alpha[i] - x[i]^2 for i in 1:length(x)];
    pop = [p; cons];

    r = ceil(Int, n*d/2);
    opt_tssos, sol_tssos, data_tssos = tssos_first(pop, variables(pop), r;
                                               TS="MD", solve=true, QUIET=true, mosek_setting=settings);

    total_time = time() - t0

    return opt_tssos, total_time
end 


### Post processing: evaluate polynomials
## Evaluate one univariate polynomial given coefficients
function eval_unipoly(cs::AbstractVector{<:Real}, x::Real)
    acc = 0.0
    for c in cs
        acc = acc * x + c
    end
    return acc
end

## Evaluate p(x) = sum_l w[l] * ∏_i f_{l,i}(x[i])
function eval_rankR_poly(
    f_coeffs_R::Vector{Vector{Vector{Float64}}},
    weights::Union{Nothing,Vector{Float64}},
    x::AbstractVector{<:Real}
)
    R = length(f_coeffs_R); n = length(x);
    w = isnothing(weights) ? ones(Float64, R) : weights;
    total = 0.0
    for l in 1:R
        prod_l = 1.0
        for i in 1:n
            prod_l *= eval_unipoly(f_coeffs_R[l][i], x[i])
        end
        total += w[l] * prod_l
    end
    return total
end

end