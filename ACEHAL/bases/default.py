params = ["elements", "cor_order", "maxdeg", "r_cut", "smoothness_prior"]

source = """using ACE1x

            elements = basis_info["elements"]
            cor_order = basis_info["cor_order"]
            maxdeg = basis_info["maxdeg"]
            r_cut = basis_info["r_cut"]
            
            B = ACE1x.ace_basis(elements = Symbol.(elements), 
                        order = cor_order, 
                        totaldegree = maxdeg, 
                        rcut = r_cut)

            B_length = length(B)
            if basis_info["smoothness_prior"][1] isa String && lowercase(basis_info["smoothness_prior"][1]) == "none"
                P_diag = nothing
            elseif basis_info["smoothness_prior"][1] isa String && basis_info["smoothness_prior"][2] isa Number && lowercase(basis_info["smoothness_prior"][1]) == "algebraic"
                P_diag = diag(smoothness_prior(B; p = basis_info["smoothness_prior"][2]))
            else
                @warn("Unkown smoothness_prior!")
                P_diag = nothing
            end
            """
