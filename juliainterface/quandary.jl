using LinearAlgebra
using Random
using Printf
using DelimitedFiles
using Plots

# Define a mutable struct to hold configuration options for the Quandary program
mutable struct QuandaryConfig
    Ne::Vector{Int}            # Number of essential energy levels per qubit
    Ng::Vector{Int}            # Number of extra guard levels per qubit
    freq01::Vector{Float64}    # 01-transition frequencies [GHz] per qubit
    selfkerr::Vector{Float64}  # Anharmonicities [GHz] per qubit
    rotfreq::Vector{Float64}   # Frequency of rotations for computational frame [GHz] per qubit
    Jkl::Vector{Float64}        # Jaynes-Cummings coupling strength [GHz] per qubit
    crosskerr::Vector{Float64}  # ZZ coupling strength [GHz] per qubit
    T1::Vector{Float64}         # Optional: T1-Decay time per qubit (invokes Lindblad solver)
    T2::Vector{Float64}         # Optional: T2-Dephasing time per qubit (invokes Lindlbad solver)
    T::Float64                  # Final time duration
    Pmin::Int                   # Number of discretization points to resolve the shortest period of the dynamics (determines <nsteps>)
    nsteps::Int                 # Number of time-discretization points (will be computed internally based on Pmin, or can be set here)
    timestepper::String         # Time-discretization scheme
    standardmodel::Bool         # Switch to use standard Hamiltonian model for superconducting qubits
    Hsys::Matrix{Float64}       # Optional: User-specified system Hamiltonian model
    Hc_re::Vector{Matrix{Float64}}  # Optional: User-specified control Hamiltonian operators for each qubit (real-parts)
    Hc_im::Vector{Matrix{Float64}}  # Optional: User-specified control Hamiltonian operators for each qubit (imaginary-parts)
    maxctrl_MHz::Vector{Float64}    # Amplitude bounds for the control pulses [MHz]
    control_enforce_BC::Bool        # Enforce that control pulses start and end at zero
    dtau::Float64                   # Spacing [ns] of Bspline basis functions (The number of Bspline basis functions will be T/dtau + 2)
    nsplines::Int                   # Number of Bspline basis functions, will be computed from T and dtau
    pcof0::Vector{Float64}          # Optional: Pass an initial control parameter vector
    pcof0_filename::String          # Optional: Load initial control parameter vector from a file
    randomize_init_ctrl::Bool       # Randomize the initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    initctrl_MHz::Vector{Float64}   # Amplitude [MHz] of initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    carrier_frequency::Vector{Vector{Float64}}  # Will be set in __post_init
    cw_amp_thres::Float64           # Threshold to ignore carrier wave frequencies whose growth rate is below this value
    cw_prox_thres::Float64          # Threshold to distinguish different carrier wave frequencies from each other
    costfunction::String            # Cost function measure: "Jtrace" or "Jfrobenius"
    targetgate::Matrix{Complex{Float64}}  # Complex target unitary in the essential level dimensions for gate optimization
    optim_target::String  # Optional: Set optimization targets, if not specified through the targetgate
    initialcondition::String        # Initial states at time t=0.0: "basis", "diagonal", "pure, 0,0,1,...", "file, /path/to/file"
    gamma_tik0::Float64             # Parameter for Tikhonov regularization term
    gamma_energy::Float64           # Parameter for integral penalty term on the control pulse energy
    gamma_dpdm::Float64             # Parameter for integral penalty term on the second state derivative
    tol_infidelity::Float64         # Optimization stopping criterion based on the infidelity
    tol_costfunc::Float64           # Optimization stopping criterion based on the objective function value
    maxiter::Int                    # Maximum number of optimization iterations
    print_frequency_iter::Int       # Output frequency for optimization iterations (Print every <x> iterations)
    usematfree::Bool                # Switch between matrix-free vs. sparse-matrix solver
    verbose::Bool                   # Switch to shut down printing to the screen
    rand_seed::Int                  # Seed for the random number generator
    _hamiltonian_filename::String   # Internal configuration. Should not be changed by the user
    _gatefilename::String           # Internal configuration. Should not be changed by the user
    popt::Vector{Float64}           # Storage for some optimization results, in case they are needed afterward
    time::Vector{Float64}           # Vector of discretized time points, could be useful for plotting the control pulses, etc.
    optim_hist::Matrix{Float64}     # Optimization history: all fields as in Quandary's output file <data>/optim_history.dat

    # Define a constructor for QuandaryConfig with default values
    function QuandaryConfig(; 
            Ne::Vector{Int}=[3], 
            Ng::Vector{Int}=[0], 
            freq01::Vector{Float64}=[4.10595], 
            selfkerr::Vector{Float64}=[0.2198], 
            rotfreq::Vector{Float64}=Vector{Float64}(), 
            Jkl::Vector{Float64}=Vector{Float64}(), 
            crosskerr::Vector{Float64}=Vector{Float64}(), 
            T1::Vector{Float64}=Vector{Float64}(), 
            T2::Vector{Float64}=Vector{Float64}(), 
            T::Float64=100.0, 
            Pmin::Int=40, 
            nsteps::Int=-1, 
            timestepper::String="IMR", 
            standardmodel::Bool=true, 
            Hsys ::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), 
            Hc_re::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), 
            Hc_im::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), 
            maxctrl_MHz::Vector{Float64}=Vector{Float64}(), 
            control_enforce_BC::Bool=true, 
            dtau::Float64=3.33, 
            nsplines::Int=-1, 
            pcof0::Vector{Float64}=Vector{Float64}(), 
            pcof0_filename::String="", 
            randomize_init_ctrl::Bool=true, 
            initctrl_MHz::Vector{Float64}=Vector{Float64}(), 
            carrier_frequency::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), 
            cw_amp_thres::Float64=1e-7, 
            cw_prox_thres::Float64=1e-2, 
            costfunction::String="Jtrace", 
            targetgate::Matrix{ComplexF64}=Matrix{ComplexF64}(), 
            optim_target::String="gate", 
            initialcondition::String="basis", 
            gamma_tik0::Float64=1e-4, 
            gamma_energy::Float64=0.01, 
            gamma_dpdm::Float64=0.01, 
            tol_infidelity::Float64=1e-3, 
            tol_costfunc::Float64=1e-3, 
            maxiter::Int=300, 
            print_frequency_iter::Int=1, 
            usematfree::Bool=true, 
            verbose::Bool=false, 
            rand_seed::Int=1234, 
            _hamiltonian_filename::String="", 
            _gatefilename::String="", 
            popt::Vector{Float64}=Vector{Float64}(), 
            time::Vector{Float64}=Vector{Float64}(), 
            optim_hist::Matrix{Float64}=Matrix{Float64}(undef, 1, 10)
            )

        # Set default values if some parameters are not provided by the user
        if length(freq01) != length(Ne)
            Ne = [2 for _ in freq01]  # Set default two-level system if Ne is not specified by the user
        end
        if length(Ng) != length(Ne)
            Ng = [0 for _ in Ne]  # Set default NO guard levels if Ng is not specified by the user
        end
        if length(selfkerr) != length(Ne)
            selfkerr = zeros(Float64, length(Ne))  # Set zero selfkerr if not specified by the user
        end
        rotfreq = isempty(rotfreq) ? freq01 : rotfreq  # Set default rotational frequency (default=freq01), unless specified by the user

        # Set default number of splines for control parameterization, unless specified by the user
        if nsplines < 0
            nsplines = max(ceil(Int, T / dtau + 2), 5)
        end

        # Set default amplitude of initial control parameters [MHz] (default = 9 MHz)
        if isempty(initctrl_MHz)
            initctrl_MHz = [9.0 for _ in Ne]
        end


        # Set default Hamiltonian operators, unless specified by the user
        if !isempty(Hsys) && !standardmodel
            standardmodel = false  # User-provided Hamiltonian operators
        else
            # Using standard Hamiltonian model
            Ntot = [sum(x) for x in zip(Ne, Ng)]
            Hsys, Hc_re, Hc_im = hamiltonians(N=Ntot, freq01=freq01, selfkerr=selfkerr, crosskerr=crosskerr, Jkl=Jkl, rotfreq=rotfreq, verbose=verbose)
            standardmodel = true
        end
        println("Hsys", Hsys)

        # Estimate the number of time steps
        nsteps = estimate_timesteps(T=T, Hsys=Hsys, Hc_re=Hc_re, Hc_im=Hc_im, maxctrl_MHz=maxctrl_MHz, Pmin=Pmin)
        if verbose
            println("Final time: $T ns, Number of timesteps: $nsteps, dt= $(T / nsteps) ns")
            println("Maximum control amplitudes: $maxctrl_MHz MHz")
        end

        # Estimate carrier wave frequencies
        carrier_frequency, _ = get_resonances(Ne=Ne, Ng=Ng, Hsys=Hsys, Hc_re=Hc_re, Hc_im=Hc_im, rotfreq=rotfreq, verbose=verbose, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, stdmodel=standardmodel)

        if verbose
            println("\n")
            for q in 1:length(Ne)
                println("System #$q Carrier frequencies (lab frame): $(rotfreq[q] .+ carrier_frequency[q])")
                println("                               (rot frame): $(carrier_frequency[q])")
            end
            println("\n")
        end

        new(Ne, Ng, freq01, selfkerr, rotfreq, Jkl, crosskerr, T1, T2, T, Pmin, nsteps, timestepper, standardmodel, Hsys, Hc_re, Hc_im, maxctrl_MHz, control_enforce_BC, dtau, nsplines, pcof0, pcof0_filename, randomize_init_ctrl, initctrl_MHz, carrier_frequency, cw_amp_thres, cw_prox_thres, costfunction, targetgate, optim_target, initialcondition, gamma_tik0, gamma_energy, gamma_dpdm, tol_infidelity, tol_costfunc, maxiter, print_frequency_iter, usematfree, verbose, rand_seed, _hamiltonian_filename, _gatefilename, popt, time, optim_hist)
    end
end

function estimate_timesteps(; T=1.0, Hsys=[], Hc_re=[], Hc_im=[], maxctrl_MHz=[], Pmin=40)
    # Get estimated control pulse amplitude
    est_ctrl_MHz = copy(maxctrl_MHz)
    if isempty(maxctrl_MHz)
        est_ctrl_MHz = fill(10.0, max(length(Hc_re), length(Hc_im)))
    end

    # Set up Hsys + maxctrl*Hcontrol
    K1 = copy(Hsys)

    for i in 1:length(Hc_re)
        est_radns = est_ctrl_MHz[i] * 2.0 * π / 1.0e3
        if !isempty(Hc_re[i])
            K1 += est_radns * Hc_re[i]
        end
    end

    for i in 1:length(Hc_im)
        est_radns = est_ctrl_MHz[i] * 2.0 * π / 1.0e3
        if !isempty(Hc_im[i])
            K1 = K1 + 1.0im * est_radns * Hc_im[i]  # Note: Julia uses `im` for the imaginary unit
        end
    end

    # Estimate time step
    eigenvalues = eigen(K1).values
    maxeig = maximum(abs.(eigenvalues))
    ctrl_fac = 1.0
    samplerate = ctrl_fac * maxeig * Pmin / (2 * π)
    nsteps = ceil(Int, T * samplerate)

    return nsteps
end



# Computes eigen decomposition and re-orders it to make the eigenvector matrix as close to the identity as possible
function eigen_and_reorder(H0::Matrix{Float64}, verbose::Bool=false)
    # Get eigenvalues and vectors and sort them in ascending order
    Ntot = size(H0, 1)
    evals, evects = eigen(H0)
    evects = Matrix(evects)  # Convert Eigen matrix to a standard matrix

    reord = sortperm(evals)
    evals = evals[reord]
    evects = evects[:, reord]

    # Find the permutation that reorders the eigenvectors closer to the identity (max. value per column on the diagonal positions)
    maxrow = zeros(Int, Ntot)
    for j in 1:Ntot
        maxrow[j] = argmax(abs.(evects[:, j]))
    end
    s_perm = sortperm(maxrow)
    evects = evects[:, s_perm]
    evals = evals[s_perm]

    # Make sure all diagonal elements are positive
    for j in 1:Ntot
        if evects[j, j] < 0.0
            evects[:, j] .= -evects[:, j]
        end
    end

    return evals, evects
end

# Computes system resonances, to be used as carrier wave frequencies
# Returns resonance frequencies in GHz and corresponding growth rates.
function get_resonances(;Ne::Vector{Int}, Ng::Vector{Int}, Hsys::Matrix{Float64}, Hc_re::Vector{Matrix{Float64}}=[], Hc_im::Vector{Matrix{Float64}}=[], rotfreq::Vector{Float64}=[], cw_amp_thres::Float64=1e-7, cw_prox_thres::Float64=1e-2, verbose::Bool=true, stdmodel::Bool=true)
    if verbose
        println("\nComputing carrier frequencies, ignoring growth rate slower than: ", cw_amp_thres, " and frequencies closer than: ", cw_prox_thres, " [GHz])")
    end

    nqubits = length(Ne)
    n = size(Hsys, 1)

    # Get eigenvalues of the system Hamiltonian (GHz)
    Hsys_evals, Utrans = eigen_and_reorder(Hsys, verbose)
    Hsys_evals = real.(Hsys_evals)  # Eigenvalues may have a small imaginary part due to numerical imprecision
    Hsys_evals ./= (2 * π)

    # Look for resonances in the symmetric and anti-symmetric control Hamiltonians for each qubit
    resonances = Vector{Vector{Float64}}(undef, nqubits)
    speed = Vector{Vector{Float64}}(undef, nqubits)

    for q in 1:nqubits
        Hsym_trans = Utrans' * Hc_re[q] * Utrans
        Hanti_trans = Utrans' * Hc_im[q] * Utrans

        resonances_a = Vector{Float64}()
        speed_a = Vector{Float64}()

        if verbose
            println("  Resonances in oscillator #", q)
        end

        for Hc_trans in (Hsym_trans, Hanti_trans)
            for i in 1:n
                for j in 1:i
                    if abs(Hc_trans[i, j]) < 1e-14
                        continue
                    end

                    delta_f = Hsys_evals[i] - Hsys_evals[j]

                    if abs(delta_f) < 1e-10
                        delta_f = 0.0
                    end

                    ids_i = map_to_oscillators(i, Ne, Ng)
                    ids_j = map_to_oscillators(j, Ne, Ng)

                    is_ess_i = all(ids_i[k] <= Ne[k] for k in 1:length(Ne))
                    is_ess_j = all(ids_j[k] <= Ne[k] for k in 1:length(Ne))

                    if is_ess_i && is_ess_j
                        if any(abs(delta_f - f) < cw_prox_thres for f in resonances_a)
                            if verbose
                                println("    Ignoring resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]), " being too close to one that already exists.")
                            end
                        elseif abs(Hc_trans[i, j]) < cw_amp_thres
                            if verbose
                                println("    Ignoring resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]), " growth rate is too slow.")
                            end
                        else
                            push!(resonances_a, delta_f)
                            push!(speed_a, abs(Hc_trans[i, j]))
                            if verbose
                                println("    Resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]))
                            end
                        end
                    end
                end
            end
        end

        resonances[q] = resonances_a
        speed[q] = speed_a
    end

    Nfreq = zeros(Int, nqubits)
    om = Vector{Vector{Float64}}(undef, nqubits)
    growth_rate = Vector{Vector{Float64}}(undef, nqubits)

    for q in 1:nqubits
        Nfreq[q] = max(1, length(resonances[q]))
        om[q] = zeros(Nfreq[q])

        if !isempty(resonances[q])
            om[q] = resonances[q]
        end

        growth_rate[q] = ones(Nfreq[q])

        if !isempty(speed[q])
            growth_rate[q] = speed[q]
        end
    end

    return om, growth_rate
end


# Lowering operator of dimension n
function lowering(n)
    return diagm(1 => sqrt.(1:n-1))
end

# Number operator of dimension n
function number(n)
    return diagm(0 => 0:n-1)
end

# Return the local energy level of each oscillator for a given global index id
function map_to_oscillators(id, Ne, Ng)
    # len(Ne) = number of subsystems
    nlevels = [Ne[i] + Ng[i] for i in 1:length(Ne)]
    localIDs = []

    index = Int(id)
    for iosc in 1:length(Ne)
        postdim = prod(nlevels[iosc+1:end])
        push!(localIDs, Int(index ÷ postdim))
        index = index % postdim
    end

    return localIDs
end


# Create standard Hamiltonian operators to model superconducting qubits
# Returns Hsys (System Hamiltonian), Hc_re (Real parts of control Hamiltonian operators), and Hc_im (Imaginary parts of control Hamiltonian operators)
function hamiltonians(;N::Vector{Int}, freq01::Vector{Float64}, selfkerr::Vector{Float64}, crosskerr::Vector{Float64}=[], Jkl::Vector{Float64}=[], rotfreq::Vector{Float64}=[], verbose::Bool=true)
    if isempty(rotfreq)
        rotfreq = zeros(Float64, length(N))
    end

    nqubits = length(N)
    @assert length(selfkerr) == nqubits
    @assert length(freq01) == nqubits
    @assert length(rotfreq) == nqubits

    n = prod(N)  # System size

    # Set up lowering operators in full dimension
    Amat = []
    for i in 1:nqubits
        ai = lowering(N[i])
        for j in 1:i-1
            ai = kron(I, ai)
        end
        for j in i+1:nqubits
            ai = kron(ai, I)
        end
        push!(Amat, ai)
    end

    # Set up system Hamiltonian: Duffing oscillators
    Hsys = zeros(Float64, n, n)
    for q in 1:nqubits
        domega_radns = 2.0 * π * (freq01[q] - rotfreq[q])
        selfkerr_radns = 2.0 * π * selfkerr[q]
        Hsys += domega_radns * Amat[q]' * Amat[q]
        Hsys -= selfkerr_radns / 2.0 * Amat[q]' * Amat[q]' * Amat[q] * Amat[q]
    end

    # Add cross kerr coupling, if given
    if !isempty(crosskerr)
        idkl = 1
        for q in 1:nqubits
            for p in q+1:nqubits
                if abs(crosskerr[idkl]) > 1e-14
                    crosskerr_radns = 2.0 * π * crosskerr[idkl]
                    Hsys -= crosskerr_radns * Amat[q]' * Amat[q] * Amat[p]' * Amat[p]
                end
                idkl += 1
            end
        end
    end

    # Add Jkl coupling term
    if !isempty(Jkl)
        idkl = 1
        for q in 1:nqubits
            for p in q+1:nqubits
                if abs(Jkl[idkl]) > 1e-14
                    Jkl_radns = 2.0 * π * Jkl[idkl]
                    Hsys += Jkl_radns * (Amat[q]' * Amat[p] + Amat[q] * Amat[p]')
                end
                idkl += 1
            end
        end
    end
    # Convert to vector of vectors 
    # Hsys = [Vector(row) for row in eachrow(Hsys)]

    # Set up control Hamiltonians
    Hc_re = [Amat[q] + Amat[q]' for q in 1:nqubits]
    Hc_im = [Amat[q] - Amat[q]' for q in 1:nqubits]

    if verbose
        println("*** $nqubits coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 =", freq01, "rot. freq =", rotfreq)
        println("Selfkerr=", selfkerr)
        println("Coupling: X-Kerr=", crosskerr, ", J-C=", Jkl)
    end

    return Hsys, Hc_re, Hc_im
end



# Define a function to dump configuration options, target gate, and Hamiltonian operators to files for Quandary use
function dump_config(self;runtype="simulation",datadir="./run_dir")

    # If given, write the target gate to file
    if length(self.targetgate) > 0
        gate_vectorized = [real(self.targetgate[:]); imag(self.targetgate[:])]
        self._gatefilename = "./targetgate.dat"
        open(joinpath(datadir, self._gatefilename), "w") do f
            for value in gate_vectorized
                @printf(f, "%20.13e\n", value)
            end
        end
        if self.verbose
            println("Target gate written to ", joinpath(datadir, self._gatefilename))
        end
    end

    # If not standard Hamiltonian model, write provided Hamiltonians to a file
    if !self.standardmodel
        # Write non-standard Hamiltonians to file
        self._hamiltonian_filename = "./hamiltonian.dat"
        open(joinpath(datadir, self._hamiltonian_filename), "w") do f
            @printf(f, "# Hsys\n")
            Hsyslist = self.Hsys[:]
            for value in Hsyslist
                @printf(f, "%20.13e\n", value)
            end

            for iosc in 1:length(self.Ne)
                # Real part, if given
                if length(self.Hc_re) >= iosc && length(self.Hc_re[iosc]) > 0
                    Hcrelist = self.Hc_re[iosc][:]
                    @printf(f, "# Oscillator %d Hc_real\n", iosc)
                    for value in Hcrelist
                        @printf(f, "%20.13e\n", value)
                    end
                end

                # Imaginary part, if given
                if length(self.Hc_im) >= iosc && length(self.Hc_im[iosc]) > 0
                    Hcimlist = self.Hc_im[iosc][:]
                    @printf(f, "# Oscillator %d Hc_imag\n", iosc)
                    for value in Hcimlist
                        @printf(f, "%20.13e\n", value)
                    end
                end
            end
        end
        if self.verbose
            println("Hamiltonian operators written to ", joinpath(datadir, self._hamiltonian_filename))
        end
    end

    # If pcof0 is given, write it to a file
    if length(self.pcof0) > 0
        self.pcof0_filename = "./pcof0.dat"
        open(joinpath(datadir, self.pcof0_filename), "w") do f
            for value in self.pcof0
                @printf(f, "%20.13e\n", value)
            end
        end
        if self.verbose
            println("Initial control parameters written to ", joinpath(datadir, self.pcof0_filename))
        end
    end

    # Prepare configuration file string for Quandary
    Nt = [self.Ne[i] + self.Ng[i] for i in 1:length(self.Ng)]
    mystring = "nlevels = " * join(Nt, ", ") * "\n"
    mystring *= "nessential = " * join(self.Ne, ", ") * "\n"
    mystring *= "ntime = " * string(self.nsteps) * "\n"
    mystring *= "dt = " * string(self.T / self.nsteps) * "\n"
    mystring *= "transfreq = " * join(self.freq01, ", ") * "\n"
    mystring *= "rotfreq = " * join(self.rotfreq, ", ") * "\n"
    mystring *= "selfkerr = " * join(self.selfkerr, ", ") * "\n"

    if length(self.crosskerr) > 0
        mystring *= "crosskerr = " * join(self.crosskerr, ", ") * "\n"
    else
        mystring *= "crosskerr = 0.0\n"
    end

    if length(self.Jkl) > 0
        mystring *= "Jkl = " * join(self.Jkl, ", ") * "\n"
    else
        mystring *= "Jkl = 0.0\n"
    end

    decay = dephase = false

    if length(self.T1) > 0
        decay = true
        mystring *= "decay_time = " * join(self.T1, ", ") * "\n"
    end

    if length(self.T2) > 0
        dephase = true
        mystring *= "dephase_time = " * join(self.T2, ", ") * "\n"
    end

    if decay && dephase
        mystring *= "collapse_type = both\n"
    elseif decay
        mystring *= "collapse_type = decay\n"
    elseif dephase
        mystring *= "collapse_type = dephase\n"
    else
        mystring *= "collapse_type = none\n"
    end

    mystring *= "initialcondition = " * string(self.initialcondition) * "\n"

    for iosc in 1:length(self.Ne)
        mystring *= "control_segments$(iosc-1) = spline, $(self.nsplines)\n"

        if length(self.pcof0_filename) > 0
            initstring = "file, " * string(self.pcof0_filename) * "\n"
        else
            initamp = self.initctrl_MHz[iosc] * 2.0 * π / 1000.0 / sqrt(2) / length(self.carrier_frequency[iosc])
            initstring = (self.randomize_init_ctrl ? "random, " : "constant, ") * string(initamp) * "\n"
        end

        mystring *= "control_initialization$(iosc-1) = " * initstring
        boundval = (length(self.maxctrl_MHz) == 0 ? 1.0e12 : self.maxctrl_MHz[iosc] * 2.0 * π / 1000.0)
        mystring *= "control_bounds$(iosc-1) = " * string(boundval) * "\n"
        mystring *= "carrier_frequency$(iosc-1) = " * join(self.carrier_frequency[iosc], ", ") * "\n"
    end

    mystring *= "control_enforceBC = " * string(self.control_enforce_BC) * "\n"

    if length(self._gatefilename) > 0
        mystring *= "optim_target = gate, fromfile, " * self._gatefilename * "\n"
    else
        mystring *= "optim_target = " * string(self.optim_target) * "\n"
    end

    mystring *= "optim_objective = " * string(self.costfunction) * "\n"
    mystring *= "gate_rot_freq = 0.0\n"
    mystring *= "optim_weights = 1.0\n"
    mystring *= "optim_atol = 1e-5\n"
    mystring *= "optim_rtol = 1e-4\n"
    mystring *= "optim_dxtol = 1e-8\n"
    mystring *= "optim_ftol = " * string(self.tol_costfunc) * "\n"
    mystring *= "optim_inftol = " * string(self.tol_infidelity) * "\n"
    mystring *= "optim_maxiter = " * string(self.maxiter) * "\n"
    mystring *= "optim_regul = " * string(self.gamma_tik0) * "\n"
    mystring *= "optim_penalty = 1.0\n"
    mystring *= "optim_penalty_param = 0.0\n"
    mystring *= "optim_penalty_dpdm = " * string(self.gamma_dpdm) * "\n"
    mystring *= "optim_penalty_energy = " * string(self.gamma_energy) * "\n"
    mystring *= "datadir = ./\n"

    for iosc in 1:length(self.Ne)
        mystring *= "output$(iosc-1) = expectedEnergy, population\n"
    end

    mystring *= "output_frequency = 1\n"
    mystring *= "optim_monitor_frequency = " * string(self.print_frequency_iter) * "\n"
    mystring *= "runtype = " * runtype * "\n"

    if length(self.Ne) < 6
        mystring *= "usematfree = " * string(self.usematfree) * "\n"
    else
        mystring *= "usematfree = false\n"
    end

    mystring *= "linearsolver_type = gmres\n"
    mystring *= "linearsolver_maxiter = 20\n"

    if !self.standardmodel
        mystring *= "hamiltonian_file = " * string(self._hamiltonian_filename) * "\n"
    end

    mystring *= "timestepper = " * string(self.timestepper) * "\n"

    # Write the configuration file
    outpath = joinpath(datadir, "config.cfg")
    open(outpath, "w") do file
        write(file, mystring)
    end

    if self.verbose
        println("Quandary config file written to: ", outpath)
    end

    return "./config.cfg"
end



# Define a function to run Quandary
function quandary_run(config::QuandaryConfig;runtype="optimization",ncores=-1,datadir="./run_dir",quandary_exec="/absolute/path/to/quandary/main",cygwin=false)

    # Create Quandary data directory
    mkpath(datadir)

    # Write the configuration to a file
    config_filename = dump_config(config, runtype=runtype, datadir=datadir)

    # Set default number of cores to prod(config.Ne), unless otherwise specified
    if ncores == -1
        ncores = prod(config.Ne)
    end

    # Execute a subprocess to run Quandary
    err = execute(runtype=runtype, ncores=ncores, config_filename=config_filename, datadir=datadir, quandary_exec=quandary_exec, verbose=config.verbose, cygwin=cygwin)

    if config.verbose
        println("Quandary data dir: ", datadir)
    end

    # Get results from Quandary output files
    timelist, pt, qt, expectedEnergy, popt, infidelity, optim_hist = get_results(Ne=config.Ne, datadir=datadir)

    # Store some results in the config file
    config.optim_hist = deepcopy(optim_hist)
    config.popt = deepcopy(popt)
    config.time = deepcopy(timelist)

    return pt, qt, expectedEnergy, infidelity
    # return 0
end


# Helper function to execute Quandary subprocess
function execute(; runtype="simulation", ncores=1, config_filename="config.cfg", datadir=".", quandary_exec="/absolute/path/to/quandary/main", verbose=false, cygwin=false)
    # Store the current working directory
    dir_org = pwd()

    # Change to the Quandary data directory
    cd(datadir)


    # Set up the run command
    if ncores > 1
        if verbose
            runcommand = `mpirun -np $ncores $quandary_exec $config_filename` 
        else
            runcommand = `mpirun -np $ncores $quandary_exec $config_filename --quiet` 
        end
    else
        if verbose
            runcommand = `$quandary_exec $config_filename` 
        else
            runcommand = `$quandary_exec $config_filename --quiet` 
        end
    end

    # Display the run command if verbose
    if verbose
        println("Running Quandary ... ")
        println(runcommand)
    end

    # Execute Quandary
    if !cygwin  # NOT on Windows through Cygwin (should work on Mac and Linux)
        # Execute Quandary without piping stdout/stderr
        if runtype != "evalcontrols"
            println("Executing '", runcommand, "' ...")
        end
        exec = pipeline(runcommand, stderr="err.log")
        result = run(exec)
        # Check the return code
        err = result.exitcode

    else
        # Execute Quandary on Windows through Cygwin
        p = open(`C:/cygwin64/bin/bash.exe`, "w")
        println(p, runcommand)
        close(p)
    end

    # Return to the previous directory
    cd(dir_org)

    if verbose
        println("DONE. pwd=\n", pwd())
    end

    return err
end


# Helper function to gather results from Quandary's output directory
function get_results(; Ne=[], datadir="./")
    dataout_dir = string(datadir, "/")

    # Get control parameters
    pcof = []
    try
        pcof = readdlm(string(dataout_dir, "/params.dat"), Float64)
        pcof = pcof[:]
    catch
        println("Can't read results from ", dataout_dir, "/params.dat")
    end

    # Get optimization history information
    # optim_hist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    optim_hist = Matrix{Float64}(undef, 1, 10)
    try
        optim_hist = readdlm(string(dataout_dir, "/optim_history.dat"))
        optim_hist = optim_hist[2:end,1:10]
    catch
        println("Can't read ", string(dataout_dir, "/optim_history.dat"))
    end

    if ndims(optim_hist) == 2
        optim_last = optim_hist[end, :]
    else
        optim_last = optim_hist
    end
    infid_last = 1.0 - optim_last[5]
    tikhonov_last = optim_last[7]
    dpdm_penalty_last = optim_last[9]

    # Get the time-evolution of the expected energy for each qubit, for each initial condition
    expectedEnergy = Vector{Vector{Any}}(undef, length(Ne))
    for iosc in 1:length(Ne)
        expectedEnergy[iosc] = Vector{Vector{Float64}}()
        for iinit in 1:prod(Ne)
            filename = string(dataout_dir, "expected", iosc-1, ".iinit", lpad(iinit-1, 4, '0'), ".dat")
            try
                x = readdlm(filename)
                push!(expectedEnergy[iosc], x[2:end, 2])  # Second column is expected energy
            catch
                println("Can't read $filename")
            end
        end
    end

    # Get the control pulses for each qubit
    pt = Vector{Vector{Float64}}()
    qt = Vector{Vector{Float64}}()
    ft = Vector{Vector{Float64}}()
    for iosc in 1:length(Ne)
        x = Matrix{Float64}(undef, 1, 4)
        try
            x = readdlm(string(dataout_dir, "./control", iosc-1, ".dat"))
        catch
            println("Can't read ", string(dataout_dir, "./control", iosc-1, ".dat"))
        end
        global timelist = x[2:end, 1]
        push!(pt, (x[2:end, 2] / (2 * pi)) * 1e3)  # Rot frame p(t), MHz
        push!(qt, (x[2:end, 3] / (2 * pi)) * 1e3)  # Rot frame q(t), MHz
        push!(ft, (x[2:end, 4] / (2 * pi)) * 1e3)  # Lab frame f(t)
    end

    return timelist, pt, qt, expectedEnergy, pcof, infid_last, optim_hist
end

##
# Plot the control pulse for all qubits 
##
function plot_pulse(Ne, timex, pt, qt)
    nrows = length(Ne)
    ncols = 1
    pl = plot(layout=(nrows, ncols), legend=:topright)
    
    for iosc in 1:length(Ne)
        pl = plot!(timex, pt[iosc], label="p(t)", color="red", subplot=iosc)
        pl = plot!(timex, qt[iosc], label="q(t)", color="blue", subplot=iosc)
        xlabel!("time (ns)")
        ylabel!("Drive strength [MHz]")
        maxp = round(maximum(abs.(pt[iosc])); digits=2)
        maxq = round(maximum(abs.(qt[iosc])); digits=2)
        # title!("Qubit $iosc\n max. drive $(round(maxp, 1)), $(round(maxq, 1)) MHz", subplot=i)
        title!("Qubit $iosc\n max. drive $maxp, $maxq MHz", subplot=iosc)
    end
    
    # pl = plot!(xticks=0:1:maximum(timex), xlims=(0.0, maximum(timex)))
    pl = plot!(legend=:topright)
    
    return pl
end

##
# Plot evolution of expected energy levels
##
function plot_expectedEnergy(Ne, timex, expectedEnergy, densitymatrix_form=false)
    nplots = prod(Ne)
    ncols = (nplots >= 4) ? 2 : 1     # 2 rows if more than 3 plots
    nrows = Int(ceil(nplots / ncols))
    figsizex = 6.4 * nrows * 0.75
    figsizey = 4.8 * nrows * 0.75
    # pl = plot(layout=(nrows, ncols), legend=:topright, size=(figsizex, figsizey))
    # pl = plot(layout=(nrows, ncols), legend=:topright)
    
    pl = []
    for iplot in 1:nplots
        iinit = !densitymatrix_form ? iplot : iplot * prod(Ne) + iplot
        # subplot!(iplot)
        
        plot_i = plot()
        for iosc in 1:length(Ne)
            label = !isempty(Ne) ? "Qubit $iosc" : ""
            plot!(timex, expectedEnergy[iosc][iinit], label=label)
        end
        
        xlabel!("time (ns)")
        ylabel!("expected energy")
        ylims!(0.0 - 1e-2, Ne[1] - 1.0 + 1e-2)
        xlims!(0.0, maximum(timex))
        
        binary_ID = (length(Ne) == 1) ? iinit : parse(Int, string(iinit, base=2))
        title!("init |$binary_ID>")
        plot_i = plot!(legend=:topright)

        push!(pl, plot_i)
    end
    
    plall = plot(pl...)
    # pl = plot!(xticks=0:1:maximum(time), yticks=0:10:ceil(maximum([maximum(expectedEnergy)...])))
    return plall
end