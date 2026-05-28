import numpy as np
import platform
from numba import jit, prange
from multiprocessing import Pool, get_context, shared_memory

# Define platform
is_windows = platform.system() == "Windows"

# Half-wave rectification function
@jit(nopython=True)
def half_exp(base, n=1):
    """Apply half-wave rectification and exponential scaling."""
    base = np.maximum(base, 0)  # Take positive part of the wave
    return base ** n

@jit(nopython=True)
def attention_rivalry_model_ORGaNICs(
    t, 
    idx,
    dt,
    ntheta,
    n_m,
    n,
    sigma,
    m,
    wh,
    tau_v1,
    tau_attention,
    tau_adaptation,
    tau_opponency,
    tau_recurrence,
    tau_binocular,
    sigma_a,
    aKernel,
    results, 
    wo, 
    wa,
    input,
    wc,
    wr,
    wr_local,
    W_rr
    ):  
    
    
    
    # Matrix to store the responses has shape: 
    # [number of masses, number of responses, number of layers, number of orientations, number of time steps]
    # Details:
    
    
        # [                         6                        ,                        6                         ,             2              ]
        
            #Responses of the model:                         |  #Layers are defined as:                         |   #Number of orientations
            #Excitatory Drive                                |  #Layer 1 = Left-eye monocular neurons           |
            #Suppressive Drive                               |  #Layer 2 = a neurons Left-eye (recurrent gain)  |
            #Estimated asymptotic firing rate                |  
            #Firing Rate                                     |  #Layer 3 = b neurons Left-eye (Input gain)      |
            #Adaptation term                                 |  #Layer 4 = Right-eye monocular neurons          |
            #Opponency                                       |  #Layer 5 = a neurons Right-eye (recurrent gain) |                        
            #                                                
            #                                                |  #Layer 6 = b neurons Right-eye (Input gain)     |
            #                                                |  #Layer 7 = Binocular-summation neurons          |
            #                                                |  #Layer 8 = a neurons Binocular (recurrent gain) | 
            #                                                |  #Layer 9 = Left-minus-right opponency neurons  |
            #                                                |  #Layer 10 = a neurons LMR opponency             |
            #                                                |  #Layer 11 = Right-minus-left opponency neurons  |    
            #                                                |  #Layer 12 = a neurons RML opponency             |
            #                                                |  #Layer 13 = Attention neurons                   |  
            #                                                |  #Layer 14 = a neurons Attention                 |

             
                                            #################               #################
                                            #################   The model   #################
                                            #################               #################
        
    #Monocular Layers 
    for lay in [0, 3]:
        #Input: for monocualar layer the input is the stimulus input strength 
        if lay == 0: #Left-eye monocular neurons
            inp = input[idx, lay, :, t]

        elif lay == 3: #Right-eye monocular neurons
            inp = input[idx, lay-2, :, t]
        # Updating excitatory drive: E
        results[idx, 0, lay, :, t] = half_exp(inp ** n_m
                                        - results[:, 5, lay, :, t-1].T.dot(wo[idx]) # opponency
                                       + results[:, 3, lay, :, t-1].T.dot(wc[idx]) # colinear facilitation
                                       + results[:, 3, lay, :, t-1].T.dot(wr[idx]) # recurrent excitation. This form as the same as that for collinear facilitation above, except that the recurrent excitation included the response of the neuron itself while the collinear facilitation was driven by responses of other neurons
                                       + results[:, 3, lay, :, t-1].T.dot(wr_local[idx]) # recurrent excitation that is purely local. In this case, each neuron received recurrent input only from itself
                                        )
    #Defining normalization pool (for monocular neurons)
    for lay in [0, 3]:
        #Compute suppressive drive: S
        results[idx, 1, lay, :, t] = np.sum(results[idx, 0, 0, :, t] + results[idx, 0, 3, :, t]) #suppressive drive: sum over all the units in normalization pool
        
        #Update response b
        results[idx, 3, lay + 2, :, t] = results[idx, 3, lay + 2, :, t-1] +  (-results[idx, 3, lay + 2, :, t-1] + (1 + results[:, 3, 12, :, t-1].T.dot(wa[idx]))) * (dt/tau_attention) 

        #Update response a
        results[idx, 3, lay + 1, :, t] = results[idx, 3, lay + 1, :, t-1] + (-results[idx, 3, lay + 1, :, t-1] + (1 - (results[idx, 1, lay, :, t] + sigma ** n_m + (wh * results[idx, 4, lay, :, t-1])**n_m))) * (dt / tau_recurrence)

        #Update response Monocular
        results[idx, 3, lay, :, t] = results[idx, 3, lay, :, t-1] + (-results[idx, 3, lay, :, t-1] + m * (half_exp(results[idx, 3, lay + 2, :, t])) * results[idx, 0, lay, :, t] + results[idx, 3, lay + 1, :, t] * results[idx, 3, lay, :, t-1]) * (dt/tau_v1)

        #Update Monocular Adaptation
        results[idx, 4, lay, :, t] = results[idx, 4, lay, :, t-1] + (-results[idx, 4, lay, :, t-1] + results[idx, 3, lay, :, t-1]) * (dt/tau_adaptation)

       
        
    #Binocular-summation and Opponency Layers    
    for lay in [6, 8, 10]:
        #Input
        if lay == 6: #Binocular-summation neurons
            inp = results[idx, 3, 0, :, t-1] + results[idx, 3, 3, :, t-1] 
        elif lay == 8: #LE-RE opponency neurons
            inp = half_exp(results[idx, 3, 0, :, t-1] - results[idx, 3, 3, :, t-1])
        elif lay == 10: #RE-LE opponency neurons
            inp = half_exp(results[idx, 3, 3, :, t-1] - results[idx, 3, 0, :, t-1])
        #Updating excitatory drive: E
        results[idx, 0, lay, :, t] = inp ** n
    for lay in [6, 8, 10]:
        #Binocular Neurons
        if lay == 6:
            #Suppressive drive
            results[idx, 1, lay, :, t] = results[idx, 0, lay, :, t]


            #Update response a
            results[idx, 3, lay+1, :, t] = results[idx, 3, lay+1, :, t-1] + (-results[idx, 3, lay+1, :, t-1] + (1 -(results[idx, 1, lay, :, t] + sigma ** n + (wh * results[idx, 4, lay, :, t-1])**n))) *  (dt / tau_recurrence)

            #Update Response Binocular
            results[idx, 3, lay, :, t] = results[idx, 3, lay, :, t-1] + (-results[idx, 3, lay, :, t-1] + results[idx, 0, lay, :, t-1] + results[idx, 3, lay+1, :, t] * (results[idx, 3, lay, :, t-1])) * (dt/tau_binocular) 

            #Update Adaptation
            results[idx, 4, lay, :, t] = results[idx, 4, lay, :, t-1] + (-results[idx, 4, lay, :, t-1] + results[idx, 3, lay, :, t-1]) * (dt/tau_adaptation)



        elif lay in (8, 10):
            #Update Suppressive drive
            results[idx, 1, lay, :, t] = np.sum(results[idx, 0, lay, :, t])

            #Update response a
            results[idx, 3, lay + 1, :, t] = results[idx, 3, lay + 1, :, t-1] + (-results[idx, 3, lay + 1, :, t-1] + (1 - (results[idx, 1, lay, :, t] + sigma**n))) * (dt/tau_recurrence)

            #Update response Opponency
            results[idx, 3, lay, :, t] = results[idx, 3, lay, :, t-1] + (-results[idx, 3, lay, :, t-1] + results[idx, 0, lay, :, t] + results[idx, 3, lay + 1, :, t] * results[idx, 3, lay, :, t-1]) * (dt/tau_opponency) 
        if lay == 8:
            results[idx, 5, 3, :, t] = np.sum(results[idx, 3, lay, :, t])#Inhibition sent to lay 1
        elif lay == 10:
            results[idx, 5, 0, :, t] = np.sum(results[idx, 3, lay, :, t])#Inhibition sent to lay 0
            
    
    inp = results[idx, 3, 6, :, t]
    

    #Excitatory drive for Attention layer
    aDrive = np.abs(inp.dot(aKernel))
    aSign = np.sign(inp.dot(aKernel))
    drive = np.sum(aDrive ** n) + sigma_a ** n
    results[idx, 0, 12, :, t] = aSign * (aDrive ** n)
    results[idx, 1, 12, :, t] = np.full(ntheta, drive, dtype=results.dtype)

    #Update response a
    results[idx, 3, 13, :, t] = results[idx, 3, 13, :, t-1] + (-results[idx, 3, 13, :, t-1] + (1 - (results[idx, 1, 12, :, t] + sigma_a**n))) * (dt/tau_recurrence)

    #Update response Attention
    results[idx, 3, 12, :, t] = results[idx, 3, 12, :, t-1] + (-results[idx, 3, 12, :, t-1] + results[idx, 0, 12, :, t] + (results[idx, 3, 13, :, t] * results[idx, 3, 12, :, t-1]))*(dt/(tau_attention))      
    return results

if is_windows:
    def simulate_mass(idx, nt, dt, ntheta, n_m, n, sigma, m, wh, tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel,
                    results_shape, results_shm_name,
                    wo, wa, input, wc, wr, wr_local, W_rr, b_0):

        shm = shared_memory.SharedMemory(name=results_shm_name)
        results = np.ndarray(results_shape, dtype=np.float64, buffer=shm.buf)

        for t in range(1, nt):
            attention_rivalry_model_ORGaNICs(t, idx, dt, ntheta, n_m, n, sigma, m, wh,
                                    tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel,
                                    results, wo, wa, input, wc, wr, wr_local, W_rr, b_0)

        shm.close()

    def run_simulation(
        nt, n_mass, dt, ntheta, n_m, n, sigma, m, wh,
        tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel, results,
        wo, wa, input, wc, wr, wr_local, W_rr, b_0, c_attention
    ):
        results_shape = results.shape
        shm = shared_memory.SharedMemory(create=True, size=results.nbytes)
        shm_results = np.ndarray(results_shape, dtype=np.float64, buffer=shm.buf)
        np.copyto(shm_results, results)

        args = [
            (
                idx, nt, dt, ntheta, n_m, n, sigma, m, wh,
                tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel,
                results_shape, shm.name,
                wo, wa, input, wc, wr, wr_local, W_rr, b_0, c_attention
            )
            for idx in range(n_mass)
        ]

        with get_context("spawn").Pool() as pool:
            pool.starmap(simulate_mass, args)

        # Copy back results and clean up shared memory
        np.copyto(results, shm_results)
        shm.close()
        shm.unlink()
        return results

else:
    @jit(nopython=True, parallel=True)
    def run_simulation(
        nt, n_mass, dt, ntheta, n_m, n, sigma, m, wh,
        tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel, results,
        wo, wa, input, wc, wr, wr_local, W_rr, b_0, c_attention):

        for t in range(1, nt):
            for idx in prange(n_mass):
                attention_rivalry_model_ORGaNICs(t, idx, dt, ntheta, n_m, n, sigma, m, wh,
                                        tau_v1, tau_attention, tau_adaptation, tau_opponency, tau_recurrence, tau_binocular, sigma_a, aKernel, results,
                                        wo, wa, input, wc, wr, wr_local, W_rr)

        return results