params = ["elements", "cor_order", "maxdeg", "r_cut", "r_in", "r_0", "pairs_r_dict"]

source = """using ASE, JuLIP, ACE1, ACE1x

            using ACE1: transformed_jacobi, transformed_jacobi_env
            using ACE1.Transforms: multitransform, transform, transform_d

            elements = basis_info["elements"]
            cor_order = basis_info["cor_order"]
            maxdeg = basis_info["maxdeg"]
            r_cut = basis_info["r_cut"]
            r_in_global = basis_info["r_in"]
            r_0_global = basis_info["r_0"]
            ##TODO julia transform_inv error when per-pair info is available
            ##TODO r_pairs = basis_info["pairs_r_dict"]
            r_pairs = Dict() ##TODO
            ##TODO

            pin = 2
            pcut = 2

            ninc = (pcut + pin) * (cor_order-1)
            maxn = maxdeg + ninc 

            transforms = Dict()
            cutoffs = Dict()

            if length(keys(r_pairs)) == 0
                trans = AgnesiTransform(; r0=r_0_global, p=2)
                ninc = (pcut + pin) * (cor_order-1)
                maxn = maxdeg + ninc 
                Pr = transformed_jacobi(maxn, trans, r_cut, r_in_global; pcut = pcut, pin = pin)
            else
                for sym_pair in keys(r_pairs)
                    transforms[Symbol.(sym_pair)] = AgnesiTransform(; r0=r_pairs[sym_pair]["r_0"], p=2)
                    cutoffs[Symbol.(sym_pair)] = (r_pairs[sym_pair]["r_in"], r_cut)
                end
                ace_transform = multitransform(transforms, cutoffs=cutoffs)
                Pr = transformed_jacobi(maxn, ace_transform; pcut = pcut, pin= pin )
            end
        
            D = ACE1.RPI.SparsePSHDegree()

            rpibasis = ACE1x.Pure2b.pure2b_basis(species =  AtomicNumber.(Symbol.(elements)),
                                       Rn=Pr, 
                                       D=D, 
                                       maxdeg=maxdeg, 
                                       order=cor_order, 
                                       delete2b = true)

            trans_r = AgnesiTransform(; r0=r_0_global, p = 2)

            envelope_r = ACE1.PolyEnvelope(2, r_0_global, r_cut)

            Jnew = transformed_jacobi_env(maxdeg, trans_r, envelope_r, r_cut)
            pair = PolyPairBasis(Jnew, Symbol.(elements))

            B = JuLIP.MLIPs.IPSuperBasis([pair, rpibasis]);
            B_length = length(B)
            P_diag = nothing
            """
