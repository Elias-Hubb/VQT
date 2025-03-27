import random
import pennylane as qml
import jax.numpy as jnp
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond
from jax._src.lax import linalg as lax_linalg
from jax import lax
from JAXPennyBroadcast import broadcast #broadcast rewritten to work with JAX
from jax._src.lax import linalg as lax_linalg
jax.config.update("jax_enable_x64", True)
import numpy as np
#

def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
def add_layer(params, N):
        '''
        Creates a layer of the quantum circuit with RZZ and Z and X rotations at some angles
        :param params: angles of two-qubit and single-qubit gates
        :param N: number of qubits to be applied to
        '''
        zz_params =jnp.array( [params[0]] * N)
        z_param = jnp.array( [params[1]] * N)
        x_param = jnp.array( [params[2]] * N)

        # ring of controlled 2-qubit gates
        broadcast(unitary=qml.MultiRZ, wires=range(N), pattern="ring", parameters=zz_params)
        # two layers of 1-qubit gates
        broadcast(unitary=qml.RX, wires=range(N), pattern="single", parameters=x_param)
        broadcast(unitary=qml.RZ, wires=range(N), pattern="single", parameters=z_param)


def create_circuit(params,N,m = 10):
            broadcast(unitary=qml.Hadamard, wires=range(N), pattern="single")
            # m layers of alternating unitaries and noise
            for l in range(m):
                add_layer(params[3 * l: 3 * (l + 1)], N)
                p_err_channel = (0.16) * params[-1]/(2*jnp.pi)
                parameters = [p_err_channel] * N
                #qml.DepolarizingChannel(p_err_channel,wires=0)
                #broadcast(unitary=qml.DepolarizingChannel, wires=range(N), pattern="single",parameters=jnp.array(parameters))
                broadcast(unitary=qml.PhaseFlip, wires=range(N), pattern="single",parameters=jnp.array(parameters))
            return qml.density_matrix(wires=range(N))#qml.density_matrix(wires=range(N))

from scipy.linalg import logm
def S(dm):
    return  - np.trace(np.matmul(dm, np.array(logm(dm))))


#Number of samples
N_it = 100
#random seed
np.random.seed(2)
params = np.random.rand(N_it,31)*2*np.pi

#qnodes
dev3 = qml.device('default.mixed', wires= 3, shots=None) 
qnode3 = jax.jit(qml.QNode(lambda params: create_circuit(params, 3), dev3,interface= "jax"))
dev4 = qml.device('default.mixed', wires= 4, shots=None) 
qnode4 = jax.jit(qml.QNode(lambda params: create_circuit(params, 4), dev4,interface= "jax"))
dev6 = qml.device('default.mixed', wires= 6, shots=None) 
qnode6 = jax.jit(qml.QNode(lambda params: create_circuit(params, 6), dev6,interface= "jax"))
dev8 = qml.device('default.mixed', wires= 8, shots=None) 
qnode8 = jax.jit(qml.QNode(lambda params: create_circuit(params, 8), dev8,interface= "jax"))
dev10 = qml.device('default.mixed', wires= 10, shots=None) 
qnode10 = jax.jit(qml.QNode(lambda params: create_circuit(params, 10), dev10,interface= "jax"))
dev12 = qml.device('default.mixed', wires= 12, shots=None) 
qnode12 = jax.jit(qml.QNode(lambda params: create_circuit(params, 12), dev12,interface= "jax"))
dev14 = qml.device('default.mixed', wires= 14, shots=None) 
qnode14 = jax.jit(qml.QNode(lambda params: create_circuit(params, 14), dev14,interface= "jax"))
node_list = [qnode3,qnode4,qnode6,qnode8,qnode10,qnode12]#

entropy_arr = np.zeros((len(node_list),N_it),dtype = complex)
for j,p in enumerate(params):
    print(j+1, " of ", N_it)
    for i,node in enumerate(node_list):
        entropy_arr[i,j] = S(node(p))
        np.save("Entropy_depolarizing_Seed_2_N100.npy",entropy_arr)
    for i,node in enumerate(node_list[:-1]):
            print("node",i+1,"of ",len(node_list))
            entropy_arr[i,j] = S(node(p))
            np.save("Entropy_phaseflip_Seed_1_.npy",entropy_arr)