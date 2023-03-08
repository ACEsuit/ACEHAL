params = ["elements", "cor_order", "maxdeg", "r_cut", "r0", "smoothness_prior"]

source = """using ACE1x, ACE1
            using ACE1.Transforms: multitransform

            elements = basis_info["elements"]
            cor_order = basis_info["cor_order"]
            maxdeg = basis_info["maxdeg"]
            r_cut = basis_info["r_cut"]
            r0 = basis_info["r0"]
            smoothness_prior_param = basis_info["smoothness_prior"]
            
            # transforms = Dict([ (s1, s2) => IdTransform() for s1 in Symbol.(elements), s2 in Symbol.(elements)] ...)

            # trans = multitransform(transforms; rin = 0.8, rcut = r_cut)

            # B = ACE1x.ace_basis(elements = Symbol.(elements), 
            #             order = cor_order, 
            #             totaldegree = maxdeg,
            #             transform = trans,
            #             poly_transform = trans,
            #             rcut = r_cut)

            Bsite = rpi_basis(species = [:C, :H, :O],
                  N = 3,      
                  maxdeg = 10,  
                  r0 = 1.0,   
                  rin = 0.5, rcut = 4.5,  
                  pin = 2)                   

            Bpair = pair_basis(species = [:C, :H, :O], 
                        r0 = 1.0, 
                    maxdeg = 3,
                            rcut = 6.0, 
                    rin = 0.0,
                            pin = 0 )  

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

            B_length = length(B)
            if isnothing(smoothness_prior_param)
                P_diag = nothing
            elseif smoothness_prior_param[1] isa String && smoothness_prior_param[2] isa Number && lowercase(smoothness_prior_param[1]) == "algebraic"
                P_diag = diag(smoothness_prior(B; p = smoothness_prior_param[2]))
            else
                throw(ArgumentError("Unknown smoothness_prior"))
            end
            """
