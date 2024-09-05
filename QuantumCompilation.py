import math
import numpy as np
import scipy.sparse as sp
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

import os
import csv
import time
import datetime
import itertools as it

import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

    
class CircuitState:
    """
    Represents a general quantum state of circuit referring to a CircuitShape.
    
    Parameters
    ------------
    shape : CircuitShape
        Shape of reference.
    steps : list[]
        List of timesteps of the ciruit. If provided, the circuit state is calculated from the timestep-string to timestep-state conversion provided by shape.
    state : np.array[]
        State of circuit. If provided, the state attribute is set to a copy of it. The state must be compatible with provided shape.

    Attributes
    -----------
    _shape : CircuitShape
        Shape of reference.
    _state : np.array[]
        Quantum state of circuit.
    """
    
    def __init__(self,shape,steps=None,state=None):  
        if state is None:
            self._shape = shape
            self._state = np.zeros((self._shape.get_n_steps()**len(steps)), dtype=np.complex128)
            index = 0
            for i in range(len(steps)):
                index += self._shape.get_step_index(steps[i]) * self._shape.get_n_steps()**(len(steps)-1-i)
            self._state[index] = 1.
        else:
            self._shape = shape
            self._state = state.copy()
        
    def get_n_qubits(self):
        return self._shape.get_n_qubits()
    def get_n_steps(self):
        return round(np.emath.logn(self._shape.get_n_steps(), self._state.shape[0]))
    def get_steps(self):
        """
        Converts the state to a list of human readable timesteps. Only works if the state is classical.
        
        Returns
        -------
        steps : list[]
        """
        if not self.is_classical(): raise Exception("The state is not classical")
        full_index = self._to_base(np.where(self._state != 0)[0][0], self._shape.get_n_steps())
        return [self._shape._steps[i] for i in full_index]
    def _to_base(self,number, base):
        result = np.zeros((self.get_n_steps()),dtype=np.intc)
        i=0
        while number:
            result[i] = number % base
            number //= base
            i+=1
        return result[::-1]
    def get_full_indexes(self):
        """
        Returns timestep indexes for every circuit in the superposition.
        
        Returns
        -------
        indexes : list[]
        """
        return [self._to_base(index, self._shape.get_n_steps()) for index in np.where(self._state != 0)[0]]
    def is_classical(self):
        return len(np.where(self._state != 0)[0]) == 1
    def get_state(self):
        return self._state.copy()
    def copy(self):
        return CircuitState(self._shape,state=self._state)
    
    def measure(self):
        """
        Collapses the state to a single-circuit state extracting in accordance with its probability distribution.
        
        Returns
        -------
        self : CircuitState
        """
        probs = np.abs(self._state)**2
        if np.abs(sum(probs) - 1) > 1e-12: raise Exception("The state is not normalized")
        measured_index = np.random.choice(range(len(self._state)), p=probs)
        self._state[:] = 0.
        self._state[measured_index] = 1.
        return self
    def normalize(self):
        """
        Normalizes the state.
        
        Returns
        -------
        self : CircuitState
        """
        self._state = self._state/np.sqrt(sum(np.abs(self._state)**2))
        return self
    
    def to_csv(self,filename):
        """
        Saves the circuit to a csv file.
        """
        with open(filename, 'w', newline='') as fout:
            writer = csv.writer(fout, delimiter=",")
            for step in self.get_steps():
                writer.writerow( step ) 
    def load_csv(filename,shape):
        """
        Static method to create a circuit from a csv file
        
        Returns
        -------
        circuit_state: CircuitState
        """
        with open(filename, 'r', newline='') as fin:
            reader = csv.reader(fin, delimiter=",")
            steps = [row for row in reader]
            return CircuitState(shape,steps=steps)
    
    def from_steps_ids(ids,shape):
        """
        Static method to create a circuit from timestep indexes referred to a shape.
        
        Returns
        -------
        circuit_state: CircuitState
        """
        steps = [shape._steps[i] for i in ids]
        return CircuitState(shape,steps=steps)
    def getID(shape,n_steps):
        """
        Static method to create an empty circuit with n_steps timesteps.
        
        Returns
        -------
        circuit_state: CircuitState
        """
        return CircuitState(shape,steps=[shape._getID() for i in range(n_steps)])
    
    def to_qiskit(self):
        """
        Returns the corresponding qiskit.circuit.QuantumCircuit
        
        Returns
        -------
        circuit: qiskit.circuit.QuantumCircuit
        """
        return CircuitState._to_qiskit_from_steps(self.get_steps())
    def _to_qiskit_from_steps(steps):
        n_qubits = len(steps[0])
        circuit = QuantumCircuit(n_qubits)
        for step in steps:
            taken = []
            for i in range(len(step)):
                    if step[i][-1] == "c":
                        for j in range(i+1,len(step)):
                            if j not in taken and step[j][:-1] == step[i][:-1] and step[j][-1] == "t":
                                if step[i][:-1] == "S":
                                    circuit.swap(i,j)
                                elif step[i][:-1] == "CZ":
                                    circuit.cz(i,j)
                                taken.append(j)
                                break #breaks when the gate is closed
        return circuit
        
    def draw(self,mpl=False):
        """
        Draws the circuit using qiskit.circuit.QuantumCircuit.draw() method
        """
        if not mpl:
            return self.to_qiskit().draw(output='text')
        else:
            return self.to_qiskit().draw("mpl")
        
    def equals(self,circuit):
        """
        Compares the state with the state of the input CircuitState. The two must refer to the same shape.
        
        Returns
        -------
        same_state: bool
        """
        return np.array_equal(self._state,circuit._state)
    def equivalent(c1,c2):
        """
        Static method to compare the unitary matrix of two input circuits. The two must have classical state and refer to the same shape.
        Returns true if the input circuits are equivalent i.e. they have the same unitary matrix (up to a phase).
        
        Returns
        -------
        equivalent: bool
        """
        U = Operator(c1.to_qiskit().compose(c2.to_qiskit().inverse())).to_matrix()
        return np.linalg.norm(U/U[0,0] - np.identity( 2**c1.get_n_qubits() )) < 1e-12
    def equivalent_from_steps(steps1,steps2):
        """
        Static method to compare the unitary matrix of two input circuits. Parameters are the two lists human-readeble timesteps.
        
        Returns
        -------
        equivalent: bool
        """
        U = Operator(CircuitState._to_qiskit_from_steps(steps1).compose(CircuitState._to_qiskit_from_steps(steps2).inverse())).to_matrix()
        return np.linalg.norm(U/U[0,0] - np.identity(  2**len(steps1[0]) )) < 1e-12
    def isID(self): 
        """
        Returns true if the circuit is equivalent to ID.
        
        Returns
        -------
        ID_equivalent: bool
        """
        U = Operator(self.to_qiskit()).to_matrix()
        return np.linalg.norm(U/U[0,0] - np.identity( 2**self.get_n_qubits() )) < 1e-12

class CircuitShape:
    """
    Manages circuit-to-quantumstate conversion used by CircuitState class. All the CircuitState of the same system must refer to the same CircuitShape object.
    Furthermore, exposes a method for compiling equivalences files.
    
    Parameters
    ------------
    n_qubits : int
        Number of qubits of the circuits.
    gates : dict{}
        Disctionary of available gates. All the gates with length n must be in a list identified by the key n. 
        It's used to calculate all the possible timesteps toghether with n_qubits.
    eq_dir : string
        Directory to store and read equivalnces files.
    infidelities : list[]
        List of infidelities of gates (indexes follow the order given by gates dictionary)
    crosstalks : matrix[][]
        Matrix of crosstalks
        
    Attributes
    -----------
    _steps: list[]
        List of all available timesteps given gates and n_qubits.
    _n_qubits : int
        Number of qubits of the circuits.
    _eq_dir :  string
        Directory to store and read equivalnces files.
    _infidelities : list[]
        List of infidelities of _steps.
    """
    def __init__(self,n_qubits,gates,eq_dir="equivalenze",infidelities=None, crosstalks=None):
        self._eq_dir = eq_dir
        self._from_gates = gates
        self._n_qubits = n_qubits
        self._steps = []
        if infidelities is None:
            self._infidelities = None
            self._find_steps(gates,[],0)
        else:
            self._infidelities = []
            if crosstalks is None:
                n_gates = sum(len(gates[key]) for key in gates.keys())
                crosstalks = np.zeros((n_gates,n_gates),dtype=np.float64)
            self._find_steps_with_infidelities(gates,[],0,[],infidelities,crosstalks)
    
    # recursively finds possible timesteps given a set of gates and a number of qubits  
    def _find_steps(self,gates,step,length):
        """
        Recursively finds possible timesteps given a set of gates and a number of qubits and add them to _steps attribute.  
        """
        if length == self._n_qubits:
            self._steps.append(step)
        else:
            for gate_length in gates.keys():
                for gate in gates[gate_length]:
                    if length+gate_length <= self._n_qubits:
                        new_step = step.copy()
                        for block in gate:
                            new_step.append(block)
                        self._find_steps(gates,new_step,length+gate_length)
    def _find_steps_with_infidelities(self,gates,step,length,included_gates,infidelities,crosstalks):
        """
        Recursively finds possible timesteps given a set of gates and a number of qubits and add them to _steps attribute, 
        also calculating and assining infidelities to each step.
        """
        if length == self._n_qubits:
            self._steps.append(step)
            infidelity = 0
            for i in range(len(included_gates)):
                infidelity += infidelities[included_gates[i]]
                for j in range(i+1,len(included_gates)):
                    infidelity += crosstalks[included_gates[i]][included_gates[j]]
            self._infidelities.append(infidelity)
        else:
            for gate_length in gates.keys():
                start_index = sum(len(gates[key]) for key in range(1,gate_length))
                i=0
                for gate in gates[gate_length]:
                    if length+gate_length <= self._n_qubits:
                        new_step = step.copy()
                        for block in gate:
                            new_step.append(block)
                        self._find_steps_with_infidelities(gates,new_step,length+gate_length,included_gates+[start_index+i],infidelities,crosstalks)
                    i+=1
    
    def state_of_step(self,step):
        """
        Returns quantum state of the single step circuit constituted by step parameter.
        
        Returns
        -------
        state : np.array[]
        """
        state = np.zeros((len(self._steps)),dtype=np.complex128)
        state[self._steps.index(step)] = 1.
        return state
    def step_of_state(self,state):
        """
        Returns the step corresponding to the single step quantum state parameter.
        
        Returns
        -------
        step : list[]
        """
        if not state.is_classical(): raise Exception("The state is not classical")
        if len(state) != len(self._steps): raise Exception("The state does not refer to a single-step circuit")
        return self._steps[np.where(state!=0)[0][0]]
    
    def get_n_qubits(self):
        return self._n_qubits
    def get_step_index(self,step):
        return self._steps.index(step)
    def get_n_steps(self):
        return len(self._steps)
    def _getID(self):
        return self._steps[0]
    def get_steps(self):
        return self._steps.copy()
    
    def print_steps(self,auto_print=True):
        out = ""
        for i in range(len(self._steps)):
            out += f"{i}: {self._steps[i]}" 
            if self._infidelities is not None:
                out += f"\t\t Inf.={self._infidelities[i]}"
            out += "\n"
        if auto_print: print(out)
        return out

    def find_equivalences(self,d_max):
        """
        Finds all the equivalences between _steps composition up to d_maxth order.
        Equivalences are saved in eq_dir directory with csv files, one per order.
        """
        redundant_eq = [ [[(i,),(i,)] for i in range(len(self._steps))] ]
        #final_eq = []

        for d in range(2,d_max+1):
            ids = list(it.product(range(len(self._steps)), repeat=d))
            redundant_eq.append([])
            #final_eq.append([])
            found = []
            
            with open(f"{self._eq_dir}/eq_d={d}.csv", 'w',newline="") as fout:
                writer = csv.writer(fout, delimiter="\t")
                writer.writerow(["n_qubits:" +str(self._n_qubits)])
                writer.writerow(["gates:"])
                for key in self._from_gates.keys():
                    writer.writerow(self._from_gates[key])
                writer.writerow(["equivalences:"])                    
                
                for i in range(len(ids)):
                    print(i,"/",len(ids))
                    redundant_eq[d-1].append([])
                    k=0
                    while k < len(found):
                        if found[k][1] == i:
                            redundant_eq[d-1][i] = [[ids[i],eq[1]] if ids[i]!=eq[1] else [ids[i],eq[0]] for eq in redundant_eq[d-1][found[k][0]]]
                            k = len(found)+1
                        else:
                            k += 1
                    if k == len(found):
                        for j in range(i+1, len(ids)):
                            # check if the rule can be obtained composing shorter eq. rules
                            d_i = 1
                            while d_i < d:
                                if [ids[i][:d_i],ids[j][:d_i]] in redundant_eq[d_i-1] or [ids[i][d-d_i:],ids[j][d-d_i:]] in redundant_eq[d_i-1]:
                                    d_i = d+1
                                else:
                                    d_i += 1
                            if d_i == d and CircuitState.equivalent_from_steps([self._steps[k] for k in ids[i]],[self._steps[k] for k in ids[j]]):
                                #final_eq[d-2].append([ids[i],ids[j]])
                                writer.writerow([step_id for step_id in ids[i]+ids[j]])
                                found.append([i,j])
                                redundant_eq[d-1][i].append([ids[i],ids[j]])
                redundant_eq[d-1] = list(it.chain(*redundant_eq[d-1]))

class CircuitOperations:
    """
    Defines operations on circuit quantum states. Expose methods to calculate Driving and Infidelity hamiltonans 
    and run the optimization simulation.
    
    Parameters
    ------------
    shape : CircuitShape
        Shape of reference.

    Attributes
    -----------
    _shape : CircuitShape
        Shape of reference.
    _hops : np.matrix[]
        Matrix containing all hop operators, e.g. |n><m| to move from state |m> to state |n>. Thtese operators are represented by scipy.sparse._csr.csr_matrix type
    _Hd : scipy.sparse._csr.csr_matrix
        Driving hamiltonian.
    _Hi : scipy.sparse._csr.csr_matrix
        Infidelity hamiltonian.
    """
    def __init__(self,shape):  
        self._shape = shape
        self._hops = np.zeros((len(shape._steps),len(shape._steps)),dtype=object)
        self._Hd = None
        self._Hi = None
        for i in range(len(shape._steps)):
            for j in range(len(shape._steps)):
                self._hops[i,j] = sp.csr_matrix( np.kron(
                    np.conj(shape.state_of_step(shape._steps[j])).reshape(-1,1),
                    shape.state_of_step(shape._steps[i])
                ), dtype=np.complex128)
    
    def get_H_eps(self,ids1,ids2,n_steps):
        """
        Calculates Hd_eps given an equivalence and the number of steps of the target circuits.

        Returns
        -------
        Hd_eps : scipy.sparse._csr.csr_matrix
        """
        id = sp.identity(len(self._hops), format='csr')
        H_eps = sp.csr_matrix((len(self._hops)**n_steps,len(self._hops)**n_steps), dtype=np.complex128)
        for i_eq in range(0,n_steps+1-len(ids1)):
            H_eps_i = sp.identity(1,format='csr') 
            for i in range(0,i_eq):
                H_eps_i = sp.kron(H_eps_i,id)
            for i in range(0,len(ids1)):
                H_eps_i = sp.kron(H_eps_i,self._hops[ids1[i],ids2[i]])
            for i in range(i_eq+len(ids1),n_steps):
                H_eps_i = sp.kron(H_eps_i,id)
            H_eps += H_eps_i
        
        return H_eps + H_eps.getH()
    def get_Hd(self, dmax, n_steps):
        """
        Calculates Hd given a max equivalence order and the number of steps of
        the target circuits, loading equivalences from _shape._eq_dir files.

        Returns
        -------
        Hd : scipy.sparse._csr.csr_matrix
        """
        Hd = sp.csr_matrix((len(self._hops)**n_steps,len(self._hops)**n_steps), dtype=np.complex128)
        for d in range(2,dmax+1):
            filename = f"{self._shape._eq_dir}/eq_d={d}.csv"
            with open(filename, 'r', newline='') as fin:
                reader = csv.reader(fin, delimiter="\t")
                #skips the header
                row = None
                while(row != "equivalences:"):
                    row = next(reader)[0]
                
                for row in reader:
                    ids1 = [int(id) for id in row[:d]]
                    ids2 = [int(id) for id in row[d:]]
                    Hd += self.get_H_eps(ids1,ids2,n_steps)
        return Hd
    def get_Hi(self, n_steps):
        """
        Calculates Hi the number of steps of the target circuits.

        Returns
        -------
        Hi : scipy.sparse._csr.csr_matrix
        """
        if self._shape._infidelities is None: raise Exception("Infidelities not provided in shape's constructor")
        Hi_step = sp.csr_matrix((self._shape.get_n_steps(),self._shape.get_n_steps()), dtype=np.complex128)
        for i in range(self._shape.get_n_steps()):
            Hi_step += self._shape._infidelities[i] * self._hops[i,i]
        
        id = sp.identity(self._shape.get_n_steps(), format='csr')
        Hi= sp.csr_matrix((self._shape.get_n_steps()**n_steps,self._shape.get_n_steps()**n_steps), dtype=np.complex128)
        for i in range(n_steps):
            Hi_i = sp.identity(1,format='csr') 
            for _ in range(0,i):
                Hi_i = sp.kron(Hi_i,id)
            Hi_i = sp.kron(Hi_i,Hi_step)
            for _ in range(0,n_steps-i-1):
                Hi_i = sp.kron(Hi_i,id)
            Hi+= Hi_i
        
        return Hi
    
    def multiply(operator,circuit_state):
        """
        Applies the operator to the state of the given circuit and returns
        a CircuitState with the result as state.

        Returns
        -------
        result_circuit : CircuitState
        """
        final_state = operator.dot(circuit_state._state)
        return CircuitState(circuit_state._shape,state=final_state)
    
    def expected_value_from_state(operator, state):
        """
        Returns the expected value of the given operator for the given state.

        Returns
        -------
        expected_value : float
        """
        expected_value = np.vdot(state,operator.dot(state))
        if expected_value.imag > 1e-12: raise Exception("The excpeted value is not real.")
        return expected_value.real
    def dev_std_from_state(operator,state):
        """
        Returns the standard deviation of the given operator for the given state.

        Returns
        -------
        std_dev : float
        """
        return np.sqrt( CircuitOperations.expected_value_from_state(operator.dot(operator),state) - CircuitOperations.expected_value_from_state(operator,state)**2 ) 
    def expected_value(operator,circuit_state):
        return CircuitOperations.expected_value_from_state(operator, circuit_state._state)
    def dev_std(operator, circuit_state):
        return CircuitOperations.dev_std_from_state(operator, circuit_state._state)
    def infidelity(self,circuit_state):
        """
        Returns infidelity expected value for the given circuit state.

        Returns
        -------
        normalized_state : np.array[]
        """
        if self._Hi is None: raise Exception("Hamiltonians not loaded")
        return CircuitOperations.expected_value(self._Hi, circuit_state).real
    def infidelity_from_statedict(state_dict):
        """
        Returns the expected value of the given operator for the given state dict, where key are infidelity
        values and entries the corresponding probability.

        Returns
        -------
        expected_value : float
        """
        return sum([inf*state_dict[inf] for inf in state_dict.keys()])
    def infidelity_err_from_statedict(state_dict):
        """
        Returns the standard deviation of the given operator for the given state dict, where key are infidelity
        values and entries the corresponding probability.

        Returns
        -------
        std_dev : float
        """
        return np.sqrt(sum([inf**2*state_dict[inf] for inf in state_dict.keys()]) - CircuitOperations.infidelity_from_statedict(state_dict)**2)
    
    def normalize(state):
        """
        Normalizes a vector state.

        Returns
        -------
        normalized_state : np.array[]
        """
        return state/np.sqrt(sum(np.abs(state)**2))
    
    def load_H(self,dmax, n_steps,get_comp_time=False):
        """
        Calculates and stores as attributes Hd and Hi. This method must be called before performing a simultaion.
        """
        if not get_comp_time:
            self._Hd = self.get_Hd(dmax,n_steps)
            self._Hi = self.get_Hi(n_steps)
        else:
            start_time_d = time.time()
            self._Hd = self.get_Hd(dmax,n_steps)
            end_time_d = time.time()
            self._Hi = self.get_Hi(n_steps)
            end_time_i = time.time()
            
            return (end_time_d-start_time_d, end_time_i-end_time_d)
    def delete_H():
        """
        Resets _Hd and _Hi attributes.
        """
        self._Hd = None
        self._Hi = None
        
    def compile(self,circuit_state,Td=1.5,Nd=4,Ti=2e3,Ni=4,thesis_sim=False,prob_treshold=None):
        if self._Hd is None or self._Hi is None: raise Exception("Hamiltonians not loaded.")
        if not circuit_state.is_classical(): raise Exception("Input state not classical.")
        
        state = circuit_state.get_state()
        i_start = np.where(state!=0)[0][0]
        
        if prob_treshold is None: prob_treshold = 1/len(state)
        
        start_time = time.time()
        evolution_d = sp.linalg.expm_multiply(-1j*self._Hd,state, start=0, stop=Td, num=Nd+1, endpoint=True)
        evolution_i = sp.linalg.expm_multiply(-1*self._Hi, evolution_d[-1], start=0, stop=Ti, num=Ni+1, endpoint=True)
        end_time = time.time()
        evolution_i = [CircuitOperations.normalize(step) for step in evolution_i]
        
        population_d = [len(np.where(np.abs(state)**2>prob_treshold)[0]) for state in evolution_d]
        population_i = [len(np.where(np.abs(state)**2>prob_treshold)[0]) for state in evolution_i]
        
        p0 = np.abs([state[i_start] for state in evolution_d])**2
        
        if thesis_sim:
            return (evolution_d,evolution_i,population_d,population_i,p0)
        else:
            return CircuitState(circuit_state._shape, state=evolution_i[-1])
        
    def thesis_simulation(self, circuit_state, dt_d, dt_i, Td_max, Ti_max, N_list, d_list,DATA,start_directory="./", dir=None, verbose=False):
        if not circuit_state.is_classical(): raise Exception("Input state not classical.")
        
        comp_times_loadH = []
        comp_times_sims = []
        bests = []
        
        Nd = int(Td_max/dt_d)
        Ni = int(Ti_max/dt_i)
        
        start_state = np.array([1],dtype=np.complex128)
        
        if dir is None:
            dir = start_directory + f"simulation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        os.makedirs(dir, exist_ok=True)
        
        FileWriter.save_descr_file(dir,circuit_state,Td_max,Ti_max,dt_d,dt_d,N_list,d_list)
        
        for N in N_list:
            start_state = np.kron(start_state,circuit_state.get_state()) 
            
            if verbose:
                print("N:",N,CircuitState(self._shape,state=start_state).draw())
            
            prob_treshold = 1/len(start_state)
            
            comp_times_loadH.append([])
            comp_times_sims.append([])
            bests.append([])
            for d in d_list:
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Loading H for N={N},d={d}...")
                comp_time_loadH = self.load_H(d,N,get_comp_time=True)
                comp_times_loadH[-1].append(comp_time_loadH)
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Loaded in " + "{:.3f}".format((comp_time_loadH[0]+comp_time_loadH[1])*1e3) + "ms.\n")
                infidelities = []
                best = (math.inf,0,0)  #inf,Td_index,Ti_index
                
                # simulation over all the possible Td,Ti
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Starting test simulations...")
                start_infidelity = CircuitOperations.expected_value_from_state(self._Hi,start_state)
                evolution_d = sp.linalg.expm_multiply(-1j*self._Hd,start_state, start=dt_d, stop=Td_max, num=Nd, endpoint=True)
                for i in range(Nd):
                    evolution_i = sp.linalg.expm_multiply(-1*self._Hi, evolution_d[i], start=dt_i, stop=Ti_max, num=Ni, endpoint=True)
                    evolution_i = [CircuitOperations.normalize(step) for step in evolution_i]

                    inf_i = [CircuitOperations.expected_value_from_state(self._Hi,state) for state in evolution_i]
                    infidelities.append(inf_i)
                    inf_min_index = np.where(inf_i == min(inf_i))[0]
                    if inf_i[inf_min_index[0]] < best[0]:    #the first one minimizes Ti
                        best = (inf_i[inf_min_index[0]],i,inf_min_index[0])
                    # elif inf_i[inf_min_index[0]] == best[0] and i+inf_min_index[0]<best[1]+best[2]: #only works if dt is the same for both Td and Ti
                    #     best = (inf_i[inf_min_index[0]],i,inf_min_index[0])
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Test simulation over.\n")
                bests[-1].append(best + (start_infidelity,))
                
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Saving infidelities...")
                FileWriter.save_heatmap(dir,N,d,Td_max,Ti_max,dt_d,dt_i,infidelities)
                del infidelities
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Infidelities saved.\n")
                
                # simulation on the best
                Td = (best[1]+1)*dt_d
                Ti = (best[2]+1)*dt_i
                if verbose: print(f"\n({time.strftime('%Hh%Mm%Ss')}) Starting simulation with best parameters (Td={Td}s,Ti={Ti}s) ...")
                N_best = 20
                evolution_d,evolution_i,population_d,population_i,p0, comp_time = self.compile(CircuitState(circuit_state._shape, state = start_state), (best[1]+1)*dt_d, N_best, (best[2]+1)*dt_i, N_best, thesis_sim=True, prob_treshold=prob_treshold)
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Simulation over.\n")
                comp_times_sims[-1].append(comp_time)
                
                if verbose: print("Optimized circuit:",CircuitState(self._shape,state=evolution_i[-1]).measure().draw())
                
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Saving best simulation...")
                evolution_dicts = []
                DATA.append(evolution_i[-1])
                for step in np.concatenate((evolution_d,evolution_i)):
                    step = sp.csr_matrix(step).tocoo()
                    state_dict = {}
                    for _,i,v in zip(step.row, step.col, step.data):
                        if np.abs(v)**2 > 1e-8:
                            s = np.zeros((step.shape[1]))
                            s[i] = 1.
                            infidelity = sum([circuit_state._shape._infidelities[timestep] for timestep in CircuitState(circuit_state._shape,state=s).get_full_indexes()[0]])
                            
                            if infidelity not in state_dict.keys():
                                state_dict[infidelity] = np.abs(v)**2
                            else:
                                state_dict[infidelity] += np.abs(v)**2
                    evolution_dicts.append(state_dict)
                FileWriter.save_best(dir,N,d,Td,Ti,Td/N_best,Ti/N_best,N_best,N_best,evolution_dicts,p0,population_d + population_i)
                if verbose: print(f"({time.strftime('%Hh%Mm%Ss')}) Best simulation saved.\n")
                del evolution_dicts,evolution_d,evolution_i,population_d,population_i,p0
        FileWriter.save_comptime_loadH(dir,N_list,d_list,comp_times_loadH)
        FileWriter.save_comptime_bestsims(dir,N_list,d_list,comp_times_sims)
        FileWriter.save_best_infidelities(dir,N_list,d_list,bests)
        
class FileWriter:
    def save_descr_file(dir,circuit_state,Td_max,Ti_max,dt_d,dt_i,N_list,d_list):
        with open(f"{dir}/simulation.txt", 'w',newline="\n") as fout:
            fout.write("Simulation of circuit " + str(circuit_state.get_steps()) + f" ({datetime.datetime.now()})\n")
            fout.write(f"Td_max={Td_max}, Ti_max={Ti_max}, dt_d={dt_d}, dt_i={dt_i}\n")
            fout.write(f"N_list = {N_list}\n")
            fout.write(f"d_list = {d_list}")
    
    def save_heatmap(dir,N,d,Td_max,Ti_max,dt_d,dt_i,infidelities):
        with open(f"{dir}/heatmap_{N}_{d}.csv", 'w',newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow([N,d])
            writer.writerow([Td_max,Ti_max,dt_d,dt_i])
            for row in infidelities:
                writer.writerow(row)
    
    def save_best(dir,N,d,Td,Ti,dt_d,dt_i,Ni,Nd,evolution_dicts,p0,population):
        with open(f"{dir}/best_{N}_{d}.csv", 'w',newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow([Td,Ti,dt_d,dt_i])
            writer.writerow([Nd,Ni])
            writer.writerow([{k: float(v) for k, v in d.items()} for d in evolution_dicts])
            writer.writerow(p0)
            writer.writerow(population)
    
    def save_comptime_loadH(dir,N_list,d_list,times):
        with open(f"{dir}/comptimes_loadH.csv", 'w',newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(N_list)
            writer.writerow(d_list)
            writer.writerow(times)
    def save_comptime_bestsims(dir,N_list,d_list,times):
        with open(f"{dir}/comptimes_bestsims.csv", 'w',newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(N_list)
            writer.writerow(d_list)
            writer.writerow(times)
    def save_best_infidelities(dir,N_list,d_list,infidelities):
        with open(f"{dir}/best_infidelities.csv", 'w',newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(N_list)
            writer.writerow(d_list)
            writer.writerow(infidelities)


class Plotter:
    def plot_evolution(dt_d,dt_i,Nd,Ni,infidelities_d,infidelities_i,dev_stds_d, dev_stds_i,p0, population):
        t_d = [dt_d*i for i in range(Nd)] 
        t_i = [t_d[-1] + dt_i*i for i in range(Ni)]
        t_tot = t_d + t_i
        
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(2, 4, height_ratios=[6, 2], width_ratios=[1,1,1,1])
        
        plt.rcParams['font.family'] = 'serif' 
        plt.rcParams['font.serif'] = ['Times New Roman'] 
        plt.rcParams['mathtext.fontset'] = 'custom'  
        plt.rcParams['mathtext.rm'] = 'Times New Roman'  
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold' 
        plt.rcParams['font.size'] = 14

        ax1 = plt.subplot(gs[0,:2])
        ax2 = plt.subplot(gs[0,2:])
        ax3 = plt.subplot(gs[1,:2])
        ax4 = plt.subplot(gs[1,2])
        ax5 = plt.subplot(gs[1,3])

        x_min_d = 0-t_d[-1]/25
        x_max_d = t_d[-1] + t_d[-1]/25
        x_min_i = t_d[-1]- (t_i[-1]-t_d[-1])/25
        x_max_i= t_i[-1] + (t_i[-1]-t_d[-1])/25
        
        y_min = 0
        y_max = max(infidelities_d+infidelities_i)+max(dev_stds_d+dev_stds_i)
        
        ax1.errorbar(t_d, infidelities_d, yerr=dev_stds_d, fmt='o', capsize=3, color=(0, 0, 179/255))
        ax1.set_xlabel(r't [1/J]')
        ax1.set_ylabel(r'Infidelity')
        ax1.set_title(r'Driving evolution')
        ax1.set_xlim(x_min_d, x_max_d)
        ax1.set_ylim(0.00, y_max)
        ax1.grid(True)
        
        ax2.errorbar(t_i, infidelities_i, yerr=dev_stds_i, fmt='o',color=(230/255, 46/255, 0),capsize=5)
        ax2.set_xlabel(r't [1/J]')
        ax2.set_ylabel(r'Infidelity')
        ax2.set_title(r'Imaginary evolution')
        ax2.set_xlim(x_min_i, x_max_i)
        ax2.set_ylim(0.00,y_max)
        ax2.grid(True)

        ax3.plot(t_d, p0,'o',color=(0, 0, 179/255), markersize=3)
        ax3.set_title(r'$p_{0}$')
        ax3.set_xlabel(r't [1/J]')
        ax3.set_yticks([0.25*i for i in range(5)])
        ax3.set_xlim(x_min_d,x_max_d)
        ax3.grid(True)
        
        ax4.plot(t_d, population[:Nd],'o',color=(0, 0, 179/255), markersize=3)
        ax4.set_title(r'Population (driving)')
        ax4.set_xlabel(r't [1/J]')
        ax4.set_yticks([0.2*i for i in range(6)])
        ax4.set_yticks(range(0,28000,7000))
        ax4.set_xlim(x_min_d,x_max_d)
        ax4.grid(True)
        
        ax5.plot(t_i, population[Nd:],'o',color=(230/255, 46/255, 0), markersize=3)
        ax5.set_title(r'Population (imaginary)')
        ax5.set_xlabel(r't [1/J]')
        ax5.set_yticks(range(0,max(population)+2, max([int(max(population)/5),2])))
        ax5.set_yticks(range(0,28000,7000))
        ax5.set_xlim(x_min_d,x_max_i)
        ax5.grid(True)
        
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((-2,1))  
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_formatter(xfmt)
        ax5.xaxis.set_major_formatter(xfmt)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    def plot_infidelities_heatmap(infidelities,Td_max,Ti_max,dt_d,dt_i,i_max,j_max,N,d):
        Nd = int(Td_max/dt_d)
        Ni = int(Ti_max/dt_i) 
        
        max_inf = 0.012
        
        plt.rcParams['font.family'] = 'serif'  
        plt.rcParams['font.serif'] = ['Times New Roman'] 
        plt.rcParams['mathtext.fontset'] = 'custom'  
        plt.rcParams['mathtext.rm'] = 'Times New Roman' 
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic' 
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold' 
        plt.rcParams['font.size'] = 13
    
        plt.imshow(np.transpose(infidelities), cmap='viridis', interpolation='nearest',origin='lower', vmin = 0.0, vmax=max_inf)

        plt.xlabel(r'$T_{D}$ [1/J]')
        plt.ylabel(r'$T_{I}$ [1/J]')
        plt.title(r'$m$ = '+str(N)+', $d_{max}$ = ' +str(d))
        

        plt.xticks(ticks=range(Nd),labels=["{:.2f}".format((i+1)*dt_d) for i in range(Nd)])
        plt.yticks(ticks=range(Ni),labels=["{:.2f}".format((i+1)*dt_i*1e-3) for i in range(Ni)])

        plt.gca().text(0, 1.03, '1e3', ha='center', va='center', rotation=0, transform= plt.gca().transAxes, fontsize=13)
        plt.colorbar(label='')
        fig = plt.gcf()
        plt.scatter([i_max], [j_max], facecolors='none', edgecolor=(230/255, 46/255, 0), s=1020, marker="s")

        

        plt.show()
        plt.rcdefaults()
        return fig
    def plot_comptime_loadH(N_list,d_list,times):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
        
        fig.suptitle("Computing time - H loading")
        
        time_graphs = []
        for i in range(len(times[0])):
            time_graphs.append([t_N[i] for t_N in times])
        
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])
        
        ax1.set_xlabel(r'N')
        ax1.set_ylabel(r'Comp. time')
        ax1.set_title(r'$H_{D}+H_{I}$')
        ax1.set_xticks(N_list)
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(ticker.NullLocator())
        
        ax2.set_xlabel(r'N')
        ax2.set_ylabel(r'Comp. time')
        ax2.set_title(r'$H_{D}$')
        ax2.set_xticks(N_list)
        ax2.grid(which='both', linestyle='--', linewidth=0.5)
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(ticker.NullLocator())
        
        ax3.set_xlabel(r'N')
        ax3.set_ylabel(r'Comp. time')
        ax3.set_title(r'$H_{I}$')
        ax3.set_xticks(N_list)
        ax3.grid(which='both', linestyle='--', linewidth=0.5)
        ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax3.yaxis.set_minor_locator(ticker.NullLocator())
        
        i=0
        colors = [(0,46/255,255/255),(139/255,0,139/255)]
        for time_graph in time_graphs:
            ax1.plot(N_list, [t[0]+t[1] for t in time_graph],'-o', color=colors[i],label=f"d = {i+2}")
            ax2.plot(N_list, [t[0] for t in time_graph], '-o',color=colors[i],label=f"d = {d_list[i]}")
            ax3.plot(N_list, [t[1] for t in time_graph], '-o',color=colors[i],label=f"d = {d_list[i]}")
            i+=1
        
        ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize='large')
        ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize='large')
        ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize='large')
        
        plt.show()
        return fig
    def plot_comptime_bestsims(N_list,d_list,times):
        time_graphs = []
        for i in range(len(d_list)):
            time_graphs.append([t_N[i] for t_N in times])
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'N')
        ax.set_ylabel(r'Comp. time [s]')
        ax.set_title(r'Optimization time')
        ax.set_xticks(N_list)
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        
        i=0
        colors = [(0,46/255,255/255),(139/255,0,139/255)]
        for time_graph in time_graphs:
            ax.plot(N_list, time_graph,'-o', color=colors[i],label=f"d = {d_list[i]}")
            i+=1
        
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize='large')
        plt.show()
        return fig
    def plot_bests(N_list,d_list,bests):
        graphs = []
        for i in range(len(d_list)):
            graphs.append([t_N[i][0]/t_N[i][3] for t_N in bests])
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'$m$')
        ax.set_ylabel(r'${\langle I {\rangle}_f} \, / \, {\langle I {\rangle}_i}$')
        ax.set_xticks(N_list)
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        
        i=0
        colors = [(0,46/255,255/255),(139/255,0,139/255)]
        for graph in graphs:
            ax.plot(N_list, graph,'--s', markerfacecolor='none',color=colors[i],label=f"d = {d_list[i]}")
            i+=1
        
        plt.show()
        return fig    
    
    def load_comptime_loadH(dir):
        with open(f"{dir}/comptimes_loadH.csv", 'r', newline='') as fin:
            reader = csv.reader(fin, delimiter="\t")
            N_list = [int(N) for N in next(reader)]
            d_list = [int(d) for d in next(reader)]
            times = [ast.literal_eval(N_times) for N_times in next(reader)]
            
            return Plotter.plot_comptime_loadH(N_list,d_list,times)    
    def load_comptime_bestsims(dir):
        with open(f"{dir}/comptimes_bestsims.csv", 'r', newline='') as fin:
            reader = csv.reader(fin, delimiter="\t")
            N_list = [int(N) for N in next(reader)]
            d_list = [int(d) for d in next(reader)]
            times = [ast.literal_eval(N_times) for N_times in next(reader)]
            
            Plotter.plot_comptime_bestsims(N_list,d_list,times)   
    def load_best_infidelities(dir):
        with open(f"{dir}/best_infidelities.csv", 'r', newline='') as fin:
            reader = csv.reader(fin, delimiter="\t")
            N_list = [int(N) for N in next(reader)]
            d_list = [int(d) for d in next(reader)]
            infidelities = [ast.literal_eval(N_times) for N_times in next(reader)]
            
            return Plotter.plot_bests(N_list,d_list,infidelities)
    def load_heatmap(dir,N,d):
        with open(f"{dir}/best_{N}_{d}.csv", 'r',newline="") as fout:
            reader = csv.reader(fout, delimiter="\t")
            params = next(reader)
            Td = float(params[0])
            Ti = float(params[1])

        with open(f"{dir}/heatmap_{N}_{d}.csv", 'r', newline='') as fin:
            reader = csv.reader(fin, delimiter="\t")
            next(reader)
            params = next(reader)
            Td_max = float(params[0])
            Ti_max = float(params[1])
            dt_d = float(params[2])
            dt_i = float(params[3])
            i_opt = int(Td/dt_d) -1
            j_opt = int(Ti/dt_i) -1
            
            infidelities = [[float(inf) for inf in row] for row in reader]
            
            return Plotter.plot_infidelities_heatmap(infidelities,Td_max,Ti_max,dt_d,dt_i,i_opt,j_opt,N,d)      
    def load_best(dir,N,d):
        with open(f"{dir}/best_{N}_{d}.csv", 'r',newline="") as fout:
            reader = csv.reader(fout, delimiter="\t")
            params = next(reader)
            Td = float(params[0])
            Ti = float(params[1])
            dt_d = float(params[2])
            dt_i = float(params[3])
            params = next(reader)
            Nd = int(float(params[0]))
            Ni = int(float(params[1]))
            
            evolution_dict = [ast.literal_eval(state_dict) for state_dict in next(reader)]
            evolution_dict_d = evolution_dict[:Nd+1]
            evolution_dict_i = evolution_dict[Nd+1:]
            
            p0 = [float(p) for p in next(reader)] 
            population = [int(p) for p in next(reader)]
            
            infidelities_d = [CircuitOperations.infidelity_from_statedict(state_dict) for state_dict in evolution_dict_d]
            infidelities_i = [CircuitOperations.infidelity_from_statedict(state_dict) for state_dict in evolution_dict_i]
            dev_stds_d = [CircuitOperations.infidelity_err_from_statedict(state_dict) for state_dict in evolution_dict_d]
            dev_stds_i = [CircuitOperations.infidelity_err_from_statedict(state_dict) for state_dict in evolution_dict_i]
            
            return Plotter.plot_evolution(dt_d,dt_i,Nd+1,Ni+1,infidelities_d,infidelities_i,dev_stds_d, dev_stds_i,p0, population)    