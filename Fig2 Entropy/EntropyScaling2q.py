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


from qubitchannelJAX import QubitChannel #
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

def CombinedPResonance(dD1,dD2,t,**kwargs):
        """Creates Resonance engineered channel .Values [dD1,dD2,t] should be in (0,1) interval are rescaled here """
        #print("dD1",dD1,"dD2",dD2,"time",t)
        
        def rates(dD): 
            """ Takes dD = delta*Delta in units of [g]. Returns the effective rates."""
            pi = jnp.pi
            kappa =  2*pi  * 0.6
            gamma =  2*pi *0.3 
            g = 2*pi *4.4
            
            O = 2*pi * 0.1
            r = gamma/kappa #Delta/delta 

            dD*=g
            d,D = jnp.sqrt(dD/r),jnp.sqrt(dD*r)
            dt = d-1j * kappa/2.
            Dt = D-1j * gamma/2.

            g_p = gamma * ((O/2)**2) * jnp.abs(dt/(dt*Dt -g**2))**2 # Equation (12) Nonunitary gates paper
            g_m = gamma * ((O/2)**2) * jnp.abs(dt/(dt*Dt -2*g**2))**2 # Equation (13) Nonunitary gates paper

            D2 = ((O/2)**2)*jnp.real(dt/(dt*Dt-g**2))#Effective stark shift of the level that is driven to an excited subspace with 2 levels
            D3 = ((O/2)**2) *jnp.real((dt*Dt -g**2)/(dt*Dt**2-2*g**2*Dt)) #Effective stark shift of the level that is driven to an excited subspace with 3 levels
            return g_p,g_m,D2,D3

        g1,g2,D00,D10 = rates(dD1*3)
        k1,k2,D11,D01 = g2,g1,D00,D10#rates(dD2*3) #g1,g2,D00,D10#
        #print("Rates:", g1,g2,k1,k2,"Time:",t)
        #print("Remaining populastion",np.exp(-g1*t))
        def Elist(k,g,t):
            """takes decay rates g,k and time t to create kraus ops of
            ___ -g-> <-k- ___ channel """
            E1 = jnp.array( [jnp.array( [0,0,] ),jnp.array( [( -1 * ( -1 + ( jnp.e )**( ( -1 * g + -1 * k ) * t ) ) * g * ( ( g + k ) )**( -1 ) )**( 1/2 ),0,] ),] )
            E2 = jnp.array( [jnp.array( [0,( -1 * ( -1 + ( jnp.e )**( ( -1 * g + -1 * k ) * t ) ) * k * ( ( g + k ) )**( -1 ) )**( 1/2 ),] ),jnp.array( [0,0,] ),] )
            E3 = jnp.array( [jnp.array( [( 2 )**( -1/2 ) * ( jnp.e )**( 1/2 * ( g + k ) * t ) * ( g + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( -1 * k + ( ( jnp.e )**( -1 * ( g + k ) * t ) * k + -1 * ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) * ( ( ( g + k ) )**( -1 ) * ( g + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( k + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k + -1 * ( ( ( ( -1 * g + ( -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( -1 * k + -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k ) ) ) )**( 2 ) + -4 * ( g * k + ( -2 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g * k + ( jnp.e )**( 2 * ( -1 * g + -1 * k ) * t ) * g * k ) ) ) )**( 1/2 ) ) ) ) ) )**( 1/2 ) * ( ( 4 * ( ( g + k ) )**( 2 ) + ( jnp.e )**( ( g + k ) * ( t ) ) * ( jnp.abs( ( -1 * g + ( ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( k + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * k + ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) ) )**( 2 ) ) )**( -1/2 ),0,] ),jnp.array( [0,( 2 )**( 1/2 ) * ( g + k ) * ( ( ( g + k ) )**( -1 ) * ( g + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( k + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k + -1 * ( ( ( ( -1 * g + ( -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( -1 * k + -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k ) ) ) )**( 2 ) + -4 * ( g * k + ( -2 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g * k + ( jnp.e )**( 2 * ( -1 * g + -1 * k ) * t ) * g * k ) ) ) )**( 1/2 ) ) ) ) ) )**( 1/2 ) * ( ( 4 * ( ( g + k ) )**( 2 ) + ( jnp.e )**( ( g + k ) * ( t ) ) * ( jnp.abs( ( -1 * g + ( ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( k + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * k + ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) ) )**( 2 ) ) )**( -1/2 ),] ),] )
            E4 = jnp.array( [jnp.array( [( 2 )**( -1/2 ) * ( jnp.e )**( 1/2 * ( g + k ) * t ) * ( g + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( -1 * k + ( ( jnp.e )**( -1 * ( g + k ) * t ) * k + ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) * ( ( ( g + k ) )**( -1 ) * ( g + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( k + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k + ( ( ( ( -1 * g + ( -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( -1 * k + -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k ) ) ) )**( 2 ) + -4 * ( g * k + ( -2 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g * k + ( jnp.e )**( 2 * ( -1 * g + -1 * k ) * t ) * g * k ) ) ) )**( 1/2 ) ) ) ) ) )**( 1/2 ) * ( ( 4 * ( ( g + k ) )**( 2 ) + ( jnp.e )**( ( g + k ) * ( t ) ) * ( jnp.abs( ( g + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( -1 * k + ( ( jnp.e )**( -1 * ( g + k ) * t ) * k + ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) ) )**( 2 ) ) )**( -1/2 ),0,] ),jnp.array( [0,( 2 )**( 1/2 ) * ( g + k ) * ( ( ( g + k ) )**( -1 ) * ( g + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( k + ( ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k + ( ( ( ( -1 * g + ( -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g + ( -1 * k + -1 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * k ) ) ) )**( 2 ) + -4 * ( g * k + ( -2 * ( jnp.e )**( ( -1 * g + -1 * k ) * t ) * g * k + ( jnp.e )**( 2 * ( -1 * g + -1 * k ) * t ) * g * k ) ) ) )**( 1/2 ) ) ) ) ) )**( 1/2 ) * ( ( 4 * ( ( g + k ) )**( 2 ) + ( jnp.e )**( ( g + k ) * ( t ) ) * ( jnp.abs( ( g + ( -1 * ( jnp.e )**( -1 * ( g + k ) * t ) * g + ( -1 * k + ( ( jnp.e )**( -1 * ( g + k ) * t ) * k + ( ( jnp.e )**( -2 * ( g + k ) * t ) * ( ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( g )**( 2 ) + ( -2 * ( 1 + ( -6 * ( jnp.e )**( ( g + k ) * t ) + ( jnp.e )**( 2 * ( g + k ) * t ) ) ) * g * k + ( ( 1 + ( jnp.e )**( ( g + k ) * t ) ) )**( 2 ) * ( k )**( 2 ) ) ) )**( 1/2 ) ) ) ) ) ) )**( 2 ) ) )**( -1/2 ),] ),] )

            Elist = [E4,E3,E2.T,E1.T]
            return Elist
        
        Elist1 = Elist(1*k1,1*g1,1e4*t)
        Elist2 = Elist(1*k2,1*g2,1e4*t)
        #Make two qubit channels from them:
        K_list = jnp.zeros((4,4,4),dtype = complex)
        Z = jnp.zeros((2,2))
        for k in range(0,4):
            K_list =K_list.at[k].set(jnp.block([[Elist1[k],Z],[Z,Elist2[k]]]) )
        QubitChannel(K_list,wires = kwargs["wires"], id ="CombinedResonance" )


def create_circuit(params,N,m = 10):
            broadcast(unitary=qml.Hadamard, wires=range(N), pattern="single")
            # m layers of alternating unitaries and noise
            for l in range(m):
                add_layer(params[4 * l: 4 * (l )+2], N)
                #Singleq
                p_err_channel = 0.16*params[-1]/(2*jnp.pi)
                parameters = [p_err_channel] * N
                broadcast(unitary=qml.PhaseFlip, wires=range(N), pattern="single",parameters=jnp.array(parameters))
                #2q
                broadcast(unitary=qml.RX, wires=range(N), pattern="single",parameters=jnp.array([4*l+3]*N))
                paramsM = jnp.array([[params[-4],params[-3],params[-2]/10]] * N) /(2*np.pi)
                
                #CombinedPResonance(sigmoid(params[-4]),sigmoid(params[-3]),sigmoid(params[-2]),wires = [0,1])
                broadcast(unitary=CombinedPResonance, wires=range(N), pattern="ring",parameters=paramsM )
                broadcast(unitary=qml.RX, wires=range(N), pattern="single",parameters=jnp.array([4*l+3]*N))
            return qml.density_matrix(wires=range(N))#qml.density_matrix(wires=range(N))

from scipy.linalg import logm
def S(dm):
    return  - np.trace(np.matmul(dm, np.array(logm(dm))))


#Number of samples
N_it = 1000
#random seed
np.random.seed(1)
params = np.random.rand(N_it,44)*2*np.pi

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
node_list = [qnode3,qnode4,qnode6,qnode8,qnode10,qnode12]

entropy_arr = np.zeros((len(node_list),N_it),dtype = complex)
for j,p in enumerate(params):
    print(j+1, " of ", N_it)
    for i,node in enumerate(node_list):
        print("node",i+1,"of ",len(node_list))
        entropy_arr[i,j] = S(node(p))
        np.save("Entropy_phaseflip_Seed_1_lownoise_lowtime_nosig_new_2q.npy",entropy_arr)
