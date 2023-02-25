params = ["elements", "cor_order", "maxdeg", "r_cut"]

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
            P_diag = nothing
            """
