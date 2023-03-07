from curses import use_default_colors
from unicodedata import decimal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg.decomp import eigh
import scipy.linalg as LA
from decimal import Decimal
import matplotlib.pyplot as plt

########
#######
"""
TODO: add Lamb shift Hamitlonian contribution.
TODO: L_D and L_R implementation. Atm. outputs zero matrices.
"""
#€##

class Lind2DNN():

    C = 299792458 # m * s^-1
    PLANCK = 4.135667662 * 10 ** (-15) # eV * s
    HBAR = PLANCK/(2*np.pi) # eV * s
    KB = 8.6173303 * 10**(-5) # eV * K^-1
    units = { "s1": 1, "ms1" : 10**-3, "us1" : 10**-6, "ns1" : 10**-9, "ps1" : 10**-12, "fs1" : 10**-15}

    def __init__(self, H_S, unit="ps1", E_reorgs=[35], w_cutoffs=[150], T_K=25, eigenbasis=True, vec="row", corr=False) -> None:
        """
        Initialize system, bath and coupling descriptors, those include:
        H_S : System Hamiltonian in cmm1 units.
        E_reorgs : List of reorganization energies of individual vibrational modes
        w_cutoffs : List of cutoff frequencies/timescales parameters of individual modes
        T_K : Bath temperature in Kelvin. It is assumed to be constant with respect to the system.
        eigenbasis :  Determines whether the computations are performed in eigen- or site basis.
        vec : Type of vectorization.
        #corr : Whether baths on different physical sites of a model system are correlated or not.
        """
        self.H_S = H_S
        self.E_reorgs = E_reorgs
        self.w_cutoffs = w_cutoffs
        self.T_K = T_K
        self.eigenbasis =  eigenbasis
        self.enes, self.wfs = eigh(H_S)
        self.dim = H_S.shape[0] #where I assume H_S is a square NxN array.
        self.vec = vec
        self.unit = unit
        #self.corr = corr

        if eigenbasis:
            self.H_S = np.dot(np.conjugate(self.wfs).T ,np.dot(H_S, self.wfs ))

        assert self.unit in self.units.keys(), """Incorrect unit choice. 
        Valid inputs are: 's1', 'ms1', 'us1', 'ns1', 'ps1', 'fs1' """

        assert vec=="row" or vec=="col", """Incorrect vectorization type. 
    Valid inputs are: 'row' for rowwise and 'col' for columnwise """

    def ene_diffs(self):
        """
        Computes all unique energy differences array of the system Hamiltonian.
        I round all differences at 8th decimal place to ensure that set can 
        correctly evaluate which ones are unique.
        """
        ene_diffs_arr = [0]
        for ene1 in self.enes:
            for ene2 in self.enes:
                ene_diffs_arr.append(round(Decimal(ene1-ene2), 8 )) 
        return map(float, (set(ene_diffs_arr)))
    
    def unitary_dynamics(self, *args):
        """
        Outputs vectorized form unitary dynamics propagator.
        *args represents any additional correction terms where each arg is a matrix of the same dimension as H_S
        """
        #uni = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)

        
        Ham = (self.H_S + sum(args)) * 100 * Lind2DNN.C

        if self.vec=="row":
            return (-1j)*(np.kron(Ham, np.eye(self.dim)) - np.kron(np.eye(self.dim), Ham.T)) * self.units[self.unit]

        elif self.vec=="col":
            return (-1j)*(np.kron(np.eye(self.dim), Ham) - np.kron(Ham.T, np.eye(self.dim))) * self.units[self.unit]
        
        #Ham = cmm1_to_freq(Ham)
        #M_U = -(1j)*(np.kron(Ham, np.eye(dim)) - np.kron(np.eye(dim), Ham.T))

        #return uni * self.units[self.unit]

    def lindblad_dissipator(self, ops, rates):
        """
        Outputs vectorized form Lindblad dissipator matrix, given operators and corresponding rates.
        """
        DISS = np.zeros((self.dim**2, self.dim**2), dtype=np.complex128)
        assert len(ops)==len(rates), "Number of operators is not equal number of rate parameters."

        if self.vec=="row":
            for op, rate in zip(ops, rates):
                DISS += rate * (np.kron(op, np.conjugate(op)) - 0.5*np.kron(np.eye(self.dim), np.dot(np.conjugate(op).T, op).T)-0.5*np.kron(np.dot(np.conjugate(op).T, op), np.eye(self.dim))) 
        elif self.vec=="col":
            for op, rate in zip(ops, rates):
                DISS += rate * (np.kron(np.conjugate(op), op) - 0.5*np.kron(np.eye(self.dim), np.dot(np.conjugate(op).T, op)) - 0.5*np.kron(np.dot(np.conjugate(op).T, op).T, np.eye(self.dim))) 
        else:
            raise Exception("""Incorrectly specified vectorization type. Valid vectorization types are:
            'row' for rowwise and 'col' for columnwise. Error occured in lindblad_dissipator() function.""")

        return DISS * self.units[self.unit]
    
    def change_basis(self, mat, basis_change=[]):
        """
        Unitary basis rotation, either as specified in the simulation object or given a manual choice inside,
        basis_change variable.
        It is implied an NxN complex unitary matrix.
        """
        if len(basis_change)>0:
            assert basis_change.shape[0] == basis_change.shape[1], "Incorrect basis change matrix shape"
            return np.dot(np.conjugate(basis_change).T ,np.dot(mat, basis_change))

        if self.eigenbasis:
            return np.dot(np.conjugate(self.wfs).T ,np.dot(mat, self.wfs))

        elif not self.eigenbasis:
            return np.dot(self.wfs ,np.dot(mat, np.conjugate(self.wfs).T))

        else:
            raise "Incorrect eigenbasis input or invalid matrix type."

    def cmm1_to_eV(self, cmm1):
        """
        Performs unit change from reciprocal centimeters(cmm1) to electron volts(eV)
        """
        return cmm1 * 100 * self.C * self.PLANCK

    def cmm1_to_freq(self, cmm1):
        """
        Performs unit change form reciprocal centimeters(cmm1) to frequency.
        """
        return (cmm1 * 100) * self.C * self.units[self.unit]

    @staticmethod
    def commutator(A,B):
        """
        Computes commutator of two matrices/operators in matrix representation.
        """
        return np.dot(A,B)-np.dot(B,A)

    @staticmethod
    def initial_density(*args, density_type="ground", dim=2):
        """Construct different initial densities depending on keyword
        ground: p_0[0,0] = 1
        mixed: p_0 = 1/dim * I 
        manual: Use args for initialization, args specify choice of populations at different sites.
        """

        if density_type == "manual":
            #TODO: Extend to non-diagonal inputs
            dens = np.diag([arg for arg in args])
            return dens/np.trace(dens)

        elif density_type == "ground":
            
            dens = np.zeros((dim,dim))
            dens[0,0] = 1
            return dens

        elif density_type == "mixed":

            dens = (1/dim) * np.eye(dim)
            return dens
        
        else:
            raise  "Incorrect choice of density_type"

    def build_lindblad_rate_ops(self, func):
        """
        Takes a function argument 'func' that relates to input bath model function with inputs types:
        frequency, index.
        frequency : Energy difference between some two states
        index : site index relating to chromophore/molecule. Can be related if local baths at different 
        molecular/chromophoric sites are different.

        constructs two arrays given parameter in object definition:
        rates : strictly non-negative values following each Lindblad operator
        I make sure they obey detailed balance condition by using appropriate bath model.
        ops : Lindblad operators of form (sum_{ij} ket{i}bra{j}) each belonging to a rate in rates array of the same index.
        """
        ops_D, ops_R, rates_D, rates_R = [], [], [], []

        ws = [val for val in Lind2DNN.ene_diffs(self)]

        """
        Optional correlated case f
        if self.corr:
        """

        for w in ws:
            for site in range(self.dim):
                Amw = np.zeros((self.dim, self.dim), dtype=np.complex128)
                #Dephasing case
                if round(Decimal(w), 8) == round(Decimal(0.0000000000), 8):
                    for M in range(self.dim):
                        Amw[M, M] += np.conjugate(self.wfs[site, M]) * self.wfs[site, M]
                    if self.eigenbasis:
                        ops_D.append(Amw)
                    else:
                        ops_D.append(np.dot(np.dot(self.wfs, Amw), np.conjugate(self.wfs.T)))
                    if len(self.E_reorgs) > 1:
                        rates_D.append(func(0, site))
                    else:
                        rates_D.append(func(0, 0))
                #Relaxation case
                else:
                    for M in range(self.dim):
                        for N in range(self.dim):
                            if (round(Decimal(w), 8) == round(Decimal(self.enes[M]-self.enes[N]), 8)): # Possibly a very bad check.
                                Amw[N,M] += np.conjugate(self.wfs[site, M]) * self.wfs[site, N]
                    if self.eigenbasis:
                        ops_R.append(Amw)
                    else:
                        ops_R.append(np.dot(np.dot(self.wfs, Amw), np.conjugate(self.wfs.T)))
                    if len(self.E_reorgs) > 1:
                        rates_R.append(func(w, site))
                    else:
                        rates_R.append(func(w, 0))
        #print("ops_R", ops_R)
        #print("rates_R", rates_R)
        return ops_D, ops_R, rates_D, rates_R 

    def einstein_bose(self, w): 
        """
        Gives Einstein-Bose population for a given energy 
        and temperature.
        """
        return 1.0/(np.exp(Lind2DNN.cmm1_to_eV(self, w)/(self.KB*self.T_K)) - 1)

    def J_ohm(self, w, ind):
        if w < 0.0:
            return 0.0
        else:
            #print("W", w, "w_cut:", self.w_cutoffs[ind])
            return (self.E_reorgs[ind]/(self.w_cutoffs[ind]*self.HBAR)) * Lind2DNN.cmm1_to_eV(self, w) * np.exp(-w/self.w_cutoffs[ind])

    def build_ohmic_bath(self, w, ind):
        """
        Computes value of spectral density induced by ohmic bath for a given frequency w.
        ind will either always be zero, if all sites in the model are subject to identical bath 
        or it will be different for each site, if even on site is subject a different bath.
        """
        if round(Decimal(w), 8) == round(Decimal(0.0), 8):
            return 2 * np.pi * self.E_reorgs[ind]/(self.w_cutoffs[ind]*self.HBAR) * self.KB * self.T_K
        else:
            return 2 * np.pi * (Lind2DNN.J_ohm(self, w, ind)*(1 + Lind2DNN.einstein_bose(self, w)) + Lind2DNN.J_ohm(self, -w, ind)*Lind2DNN.einstein_bose(self,-w)) 

    @staticmethod
    def get_density_plot(propagator, density, tmin=0, tmax=50, ts=100):
        """
        *args is any 1 x N^2 array representing appropriately vectorized density matrix.
        **kwargs requires specification of:
        -tmin: starting time
        -tmax: ending time
        -ts: number of timesteps
        -plots: tuples pointing to density matrix elements that we want to plot.

        Outputs a times array and a list with corresponding density matrices at those respective times.
        """
        assert density.shape[0] == propagator.shape[0], "Incorrect input length for *args"

        size = int(np.sqrt(len(density)))
        for t in np.linspace(tmin, tmax, ts):
            yield t, np.dot(LA.expm(propagator*t), density.T).reshape(size,size)


if __name__ == "__main__":
    #h = np.array([[0,-200],[-200, 800]])
    h = np.array([[0,-4, -8],[-4, 700, -60],[-8, -60, 800]], dtype=np.complex128)
    

    sim = Lind2DNN(h, E_reorgs=[35], vec="col", unit="ps1", eigenbasis=False)  
    U = sim.unitary_dynamics()
    rho_0 = Lind2DNN.initial_density(0, 0, 1, density_type="manual")
    rho_0 = rho_0.flatten()
    OD, OR, DR, RR = sim.build_lindblad_rate_ops(sim.build_ohmic_bath)
    L_D = sim.lindblad_dissipator(OD, DR)
    L_R = sim.lindblad_dissipator(OR, RR)

    #print("L_R", L_R)
    #print("L_D", L_D)

    L_superop = U + L_D + L_R


    L_test = U + L_D + L_R

    w ,v = LA.eig(L_test)
    xs = w.real
    ys = w.imag
    """
    plt.plot(xs, ys, ".")
    plt.vlines(x=0, ymin=min(ys), ymax= max(ys), color= "c", ls="--")
    plt.hlines(y = 0, xmin = min(xs), xmax= max(xs), color= "c", ls="--")
    plt.show()
    plt.close()
    """
    """
    fig, axes = plt.subplot(9,1, figsize=(5, 45))
    for i in range(9):
        mode_0 = v[:,i]
        abss =[abs(val) for val in mode_0]
        reals = [val.real for val in mode_0]
        abss_m = [-1*val for val in abss]
        axes[i].plot(abss, ls= "-",color='y', label="")
        axes[i].plot(abss_m, 'y')
        axes[i].plot(reals, "r--")
    """



    times, rhos = [], []
    for t, p in Lind2DNN.get_density_plot(L_superop, rho_0, tmin=0.0, tmax=2, ts=100):
        times.append(t)
        rhos.append(p) # or other choice of element

    #rhos = [np.dot( np.conjugate(sim.wfs).T ,np.dot( rho , sim.wfs)) for rho in rhos]
    rho1 = [rho[0,0].real for rho in rhos]
    rho2 = [rho[1,1].real for rho in rhos]
    rho3 = [rho[2,2].real for rho in rhos]

    ys = [LA.norm(LA.expm(L_superop*t),2) for t in times]
    #fig, axes = plt.subplots(1,2)
    plt.plot(times,ys, label="Lind 2-norm plot")
    
    #plt.plot(times, rho1, label="00") 
    #axes[0].plot(times, np.gradient(rho1), labeL="11") 
    #plt.plot(times, rho2, label="11") 
    #plt.plot(times, rho3, label="2") 
    
    plt.legend()
    plt.show()
    #print("dens 11: ", times[np.argmax(np.gradient(rho1))], "L_norm: ", times[np.argmax(ys)] )
