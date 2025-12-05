using DataFrames, CSV, PrettyTables, Printf

include(joinpath(@__DIR__, "LRPOP.jl"))  
import .LRPOP

# Make a wide DataFrame: first column r, then one column per n
function wide_df(M::AbstractMatrix, r_vals::AbstractVector{<:Integer}, n_vals::AbstractVector{<:Integer})
    @assert size(M,1) == length(r_vals) && size(M,2) == length(n_vals)
    df = DataFrame(r = Int.(r_vals))
    for (j,n) in enumerate(n_vals)
        df[!, Symbol("n=$(n)")] = Float64.(M[:, j])
    end
    return df
end

# Single-row DataFrame for TSSOS (no relaxation order, r, axis)
function tssos_row_df(vals::AbstractVector, n_vals::AbstractVector{<:Integer})
    @assert length(vals) == length(n_vals)
    df = DataFrame()
    for (j,n) in enumerate(n_vals)
        df[!, Symbol("n=$(n)")] = [Float64(vals[j])]
    end
    return df
end

# Function to run experiments comparing LRPOP and TSSOS
function lrpop_vs_tssos(R::Int, d::Int, n_max::Int)
    
        n = 2;
        r = ceil(Int, (d+1)/2);
        weights = ones(R);
        alpha = ones(Float64, n);
        alpha_n = fill(1.0/n, n);

        # Initialize solvers to study running times more accurately afterwards
        f_coeffs_R = LRPOP.random_rank_components(R, n, d; alpha);
        full_poly = LRPOP.build_full_poly(f_coeffs_R, weights);
        opt_LRPOP, t_LRPOP = LRPOP.solve_LRPOP(R, n, d, r; alpha, weights, f_coeffs_R);
        opt_TSSOS, t_TSSOS = LRPOP.solve_TSSOS(f_coeffs_R, weights, n, d);

        # Set limits for the two types of graphs
        r_min = ceil(Int, (d+1)/2);
        r_max = r_min + 1;

        # Matrices of data for optimal values and runtimes
        M_LRopts = fill(NaN, r_max - r_min + 1, n_max-1)
        M_LRt    = fill(NaN, r_max - r_min + 1, n_max-1)

        M_TSopts = fill(NaN, n_max-1)
        M_TSt    = fill(NaN, n_max-1) 

        # LRPOP data
        for n in 2:n_max
                print(n)
                local alpha = ones(Float64, n);
                local f_coeffs_R = LRPOP.random_rank_components(R, n, d; alpha);
                local t_TSSOS = 0;

                for r in r_min:r_max
                        local opt_LRPOP, t_LRPOP, _ = LRPOP.solve_LRPOP(R, n, d, r; alpha, weights, f_coeffs_R);
                        M_LRopts[r-r_min+1,n-1] = opt_LRPOP;
                        M_LRt[r-r_min+1,n-1] = t_LRPOP;
                end
        end

        ## TSSOS data
        for n in 2:n_max
                local alpha = ones(Float64, n);
                local f_coeffs_R = LRPOP.random_rank_components(R, n, d; alpha);
                local t_TSSOS = 0;
                local opt_TSSOS, t_TSSOS = LRPOP.solve_TSSOS(f_coeffs_R, weights, n, d, alpha);
                M_TSopts[n-1] = opt_TSSOS;
                M_TSt[n-1] = t_TSSOS;

                if t_TSSOS > 300.0
                        break
                end
        end

        dir_path = "Numerics/Results_R$(R)_d$(d)"
    
        # Create the directory to save the CSVs
        function save_results_csv(filename::AbstractString, df::DataFrame)
                # Ensure the directory exists (mkpath is recursive)
                mkpath(dir_path)
                full_path = joinpath(dir_path, filename)
                CSV.write(full_path, df)
        end

        # Ranges
        n_vals = collect(2:n_max)
        r_vals = collect(r_min:r_max)

        # Build the wide DataFrames
        df_opt = wide_df(M_LRopts, r_vals, n_vals)
        df_time = wide_df(M_LRt, r_vals, n_vals)

        df_opt_tssos = tssos_row_df(M_TSopts, n_vals)
        df_time_tssos = tssos_row_df(M_TSt, n_vals)

        # Save CSVs 
        save_results_csv("opt_lrpop.csv", df_opt)
        save_results_csv("time_lrpop.csv", df_time)

        save_results_csv("opt_tssos.csv", df_opt_tssos)
        save_results_csv("time_tssos.csv", df_time_tssos)
end

lrpop_vs_tssos(2,2,3);
