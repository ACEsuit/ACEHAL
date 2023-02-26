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
            if typeof(basis_info["smoothness_prior"]) == String && lowercase(basis_info["smoothness_prior"]) == "none"
                P_diag = nothing
            elseif typeof(basis_info["smoothness_prior"][1]) == String && lowercase(basis_info["smoothness_prior"][1]) == "algebraic"
                P_diag = diag(smoothness_prior(B; p = basis_info["smoothness_prior"][2]))
            end
            """
