# NOTE: agnesi_q determines where (in relation to r_0) transform transitions from flexible to
# envelope taking over.
params = ["elements", "cor_order", "maxdeg_ACE", "maxdeg_pair", "r_cut_ACE", "r_cut_pair",
          "r_0", "agnesi_q"]

source = """# Careful: This is with ACE1.jl@0.10.6

            using ACE1, JuLIP, ACE1x, ACE1pack, StaticArrays ##NB , Plots 
            using ACE1: transformed_jacobi, transformed_jacobi_env
            using ACE1.Transforms: agnesi_transform, AffineT
            using Statistics: mean

            ## ------- basic parameters 

            elements = Symbol.(basis_info["elements"])
            r0 = basis_info["r_0"]
            pin = 2
            pcut = 2
            cor_order = basis_info["cor_order"]
            maxdeg = basis_info["maxdeg_ACE"]
            maxdeg_pair = basis_info["maxdeg_pair"]
            r_cut_ACE = basis_info["r_cut_ACE"]
            r_cut_pair = basis_info["r_cut_pair"]

            ## ------ Many-Body Basis 
            # transform for many-body basis. Note that rin = 0.0. If this 
            # doesn't work, then I need to make some changes to ACE1.jl
            # we should try with other parameters, but for now most combinations won't 
            # work due to a gap in the implementation of the Agnesi(p, q) transform.
            agnesi_p, agnesi_q = 2, basis_info["agnesi_q"]
            trans_ace = agnesi_transform(r0, agnesi_p, agnesi_q)
            ninc = (pcut + pin) * (cor_order-1)
            maxn = maxdeg + ninc 
            r_in = 0.0 # please don't change this for now
            Pr_ace = transformed_jacobi(maxn, trans_ace, r_cut_ACE, r_in; pcut = pin, pin = pin)
            rpibasis = ACE1x.Pure2b.pure2b_basis(species = AtomicNumber.(elements),
                                       Rn=Pr_ace, 
                                       D=ACE1.RPI.SparsePSHDegree(),
                                       maxdeg=maxdeg, 
                                       order=cor_order, 
                                       delete2b = true)

            ## --------- Plot the transforms and transformed bases 
            # use this cell for plotting the transforms and the radial bases 
            # to visualize the effect of different parameters

            # rp = range(0.0, r_cut_ACE, length=200)
            # plt1 = plot(rp, trans_ace.(rp), label = "Agnesi($agnesi_p, $agnesi_q)")
            # vline!(plt1, [r0,], label = "r0")

            # _Rn(r, n) = ACE1.evaluate(Pr_ace, r)[n]
            # plt2 = plot(rp, _Rn.(rp, 1), label = "R1")
            # plot!(plt2, rp, _Rn.(rp, 2), label = "R2")
            # plot!(plt2, rp, _Rn.(rp, 3), label = "R3")
            # plot!(plt2, rp, _Rn.(rp, 4), label = "R4")
            # vline!(plt2, [r0,], label = "r0")
            # plot(plt1, plt2, layout = (2,1), size = (400, 600))

            ## -------- Pair Basis 

            # the transform for the radial basis should be ok with (1, 4)
            trans_pair = agnesi_transform(r0, 1, 4)
            # MUST use this envelope please. But explore 1 vs 2 vs 3! This will give the 
            # asymptotic r^(-a) of the potential as r -> 0.
            envelope_pair = ACE1.PolyEnvelope(2, r0, r_cut_pair)
            Pr_pair = transformed_jacobi_env(maxdeg_pair, trans_pair, envelope_pair, r_cut_pair)

            pair = pair_basis(species = Symbol.(elements),
                   r0 = 0.0,
                   rbasis = Pr_pair, 
                   rin = 0.0,
                   pin = 0 )

            B = JuLIP.MLIPs.IPSuperBasis([pair, rpibasis]);
            B_length = length(B)
            P_diag = 1 .+ vcat(ACE1.scaling.(B.BB, 3)...)
            """
