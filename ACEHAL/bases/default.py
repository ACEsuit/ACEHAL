params = ["elements", "cor_order", "maxdeg", "r_cut", "smoothness_prior"]

source = """using ACE1x

            elements = basis_info["elements"]
            cor_order = basis_info["cor_order"]
            maxdeg = basis_info["maxdeg"]
            r_cut = basis_info["r_cut"]
            smoothness_prior_param = basis_info["smoothness_prior"]
            
            B = ACE1x.ace_basis(elements = Symbol.(elements), 
                        order = cor_order, 
                        totaldegree = maxdeg, 
                        rcut = r_cut)

            B_length = length(B)
            if isnothing(smoothness_prior_param)
                P_diag = nothing
            elseif smoothness_prior_param[1] isa String && smoothness_prior_param[2] isa Number && lowercase(smoothness_prior_param[1]) == "algebraic"
                P_diag = diag(smoothness_prior(B; p = smoothness_prior_param[2]))
            else
                throw(ArgumentError("Unknown smoothness_prior"))
            end
            """
