# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:10:12 2023

@author: Mark Kalsbeek


Validation case based on data from:
"Poisson's Ratio of Low-Temperature PECVD Silicon Nitride Thin Films" Walmsley et al, 2007
http://ieeexplore.ieee.org/document/4276823/

In short: They created square and rectangular membranes from low and high stress SiN. Then inflated them using gass pressure and reported measured deflections.

Available data:

Test data:
- crosssectional bulge profiles through the center of a square membrane and a rectangular membrane under 300 Pa of pressure for a 305-nm-thick PECVD SiNxHy film that was deposited at 205 ◦C
- Measured relationship between applied pressure and bulge height for square and rectangular PECVD SiNxHy membranes that were deposited at (a) 125 ◦C and (b) 205 ◦C

Curve-fitting parameters for the pressure-bulge height values:

| SiN:H_y Deposition Temp. (°C) | A1/s (kPa/µm) | A2/s (kPa/µm^3) | A1/r (kPa/µm) | A2/r (kPa/µm^3) |
|-------------------------------|---------------|-----------------|---------------|-----------------|
| 125                           | 1.094x10^-2   | 1.630x10^-4     | 6.462x10^-3   | 9.490x10^-5     |
| 205                           | 3.426x10^-2   | 1.259x10^-4     | 2.014x10^-3   | 7.291x10^-5     |

Resulting material properties:
| SiN:H_y Deposition Temperature (°C) | E (GPa)   | σ0 (MPa)  |
|-------------------------------------|-----------|-----------|
| 125                                 | 79 ± 4    | 5.1 ± 1.2 |
| 205                                 | 151 ± 6   | 29 ± 3    |

Geometrical information:
    


Note:
    There is an inconsistency in the test data. 
    The bulge in the square membrane case measures 6.8 um in inkscape.
    Plugging that into the provided fitted curve I find a reported pressure of 273 Pa.
    While the figure claims that this was for a pressure of 300 Pa. 
    This is a 9.0% discrepancy. 
    The other way around, plugging in 300 Pa I find an expected bulge-height of 7.32 um.
    This is a 7.6% discrepancy.
    
Initial stress: 
    Due to differential coefficients of thermal expansion the SiN has internal stresses
    To r epresent them we will pre-stretch the mesh using the stress_self command
    Because it is a bi-axial stress state we will use strain = sigma_0 / E + v sigma_0 / E 
    Where v is Poissons' ratio
"""
import logging  

import numpy as np
import matplotlib.pyplot as plt

from src.particleSystem.ParticleSystem import ParticleSystem 
from src.Sim.simulations import Simulate_airbag 
import src.Mesh.mesh_functions as MF

curve_fitting_parameters = {
    "125": {
        "A1/s": 0.01094,
        "A2/s": 0.000163,
        "A1/r": 0.006462,
        "A2/r": 9.49e-05
    },
    "205": {
        "A1/s": 0.03426,
        "A2/s": 0.0001259,
        "A1/r": 0.002014,
        "A2/r": 7.291e-05
    }
}

material_properties = {
    "125": {
        "E [GPa]": 79,
        "Sigma_0 [MPa]": 5.1
    },
    "205": {
        "E [GPa]": 151,
        "Sigma_0 [MPa]": 29
    },
    'rho': 3.17e3 # [kg/m3]
}

height_bulge = np.poly1d(
    [curve_fitting_parameters['205']["A2/s"],
     0, 
     curve_fitting_parameters['205']["A1/s"],
     0])


params = {
    # model parameters
    "k": 250,  # [N/m]   spring stiffness
    "k_d": 250,  # [N/m] spring stiffness for diagonal elements
    "c": 10,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node
    
    # simulation settings
    "dt": 1e-5,  # [s]       simulation timestep
#    'adaptive_timestepping':0, # [m] max distance traversed per timestep
    "t_steps": 1e4,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    # !!! Check if this may be preventing it, 1e-5 tol on 1e-9 displacements
    "rel_tol": 1e-10,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e2,  # [-]       maximum number of iterations]
    "convergence_threshold": 1e-25, # [-]
    "min_iterations": 10, # [-]
    
    # sim specific settigns
    "pressure": 3e2, # [Pa] inflation pressure
    
    # Geometric parameters
    "length": 1900 * 1e-6, # [m]
    "width": 1900 * 1e-6, # [m]
    "thickness": 305e-9 # [m]
    }

n_segments = 25
poisson_ratio = -0.25
diagonal_spring_ratio = 0.306228 # Coefficient determined for this mesh shape to provide poissons ratio of 0.25

params['k'] = material_properties['205']["E [GPa]"]*1e9*params['thickness']*n_segments / (n_segments+1+ 2*n_segments*diagonal_spring_ratio/np.sqrt(2))
params['k_d'] = params['k']*diagonal_spring_ratio 


params['edge_length']= params['width']/n_segments
A_cross = params['thickness'] * params['edge_length']
params['adaptive_timestepping'] = params['edge_length']/100

# Initialize the particle system
initial_conditions, connections = MF.mesh_airbag_square_cross(params['length'],
                                                              params['edge_length'],
                                                              params = params  ,
                                                              width = params['width'],
                                                              noncompressive = False,
                                                              sparse=False)

PS = ParticleSystem(connections, initial_conditions,params)

# Now we do some fine-tuning on the particle system
# Assign the correct particle masses
PS.calculate_correct_masses(params['thickness'],material_properties['rho'])

# Calculate biaxial stiffness coefficient and apply pre_stress
params['E_biaxial'] = material_properties['205']["E [GPa]"]/(1 + poisson_ratio)
E = material_properties['205']["E [GPa]"]*1e9
strain_0 = (material_properties['205']["Sigma_0 [MPa]"]*1e6) /E#(params['E_biaxial']*1e9)
PS.stress_self(1/(1+strain_0))

# Checking if pre-stess is correct
f = PS._ParticleSystem__one_d_force_vector()
f = f.reshape((len(f)//3,3))
# Due to stress concentrations at edges will only look at a centered segment
# There is still a question of what the correct way of looking at stress is here
# Is it the stress felt in the center, or the average stress in the whole edge?
# Do we see the stress concentrations at the edges as meaningfull or not?
f_center = f[int(n_segments/2)]
sigma_0 = f_center[1] / (params['thickness']*params['edge_length'])
print(f'Applied pre-stress is {sigma_0/1e6:.2f} [MPa]')


#%% 
# Altering boundary conditions
for particle in PS.particles:
    x = particle.x
    if particle.fixed:
    #if x[0] == params['width']/2 or x[1] == params['width']/2:
        particle.set_fixed(True)
    if not particle.fixed and False: # adding some noise to help get it startd
        particle.x[2] += (np.random.random()-0.5) * 5e-7

Sim = Simulate_airbag(PS, params)

#%% 
def sweep_pressure():
    pressures = np.linspace(0.1, 0.9, 17)
    z_sim_hist = []
    z_exp_hist = []
    print('='*60)
    print('Starting Simulation...')
    for pressure in pressures:
        print('='*60)
        params['pressure'] = pressure*1e3
        Sim = Simulate_airbag(PS, params)
        Sim.run_simulation(plotframes=0, 
                           plot_whole_bag = False,
                           printframes=0,
                           simulation_function = 'kinetic_damping',
                           both_sides=False)
        x, v = PS.x_v_current_3D
        z_max = x[:,2].max()
        z_sim_hist.append(z_max)
        
        z_exp = (height_bulge-pressure).roots.real
        z_exp = z_exp[z_exp>0]
        z_exp_hist.append(z_exp)
        
        print('-'*60)
        print(f'{pressure=}[kPa]')
        print(f'Expected height:\t{z_exp[0]*1e-6:.2e}')
        print(f'Simulated height:\t{z_max:.2e}')
        print(f'Error: \t\t\t\t{(z_max*1e6-z_exp[0])/z_exp[0]*100:.1f}%')
        print('='*60)
        print('\n')
    z_sim_hist = np.array(z_sim_hist) * 1e6
    plt.plot(pressures, z_sim_hist, label='Simulated')
    plt.plot(pressures, z_exp_hist, label='Expected')
    plt.title('Simulated verus expected behaviour')
    plt.xlabel('pressure [kPa]')
    plt.ylabel('Bulge Height [$\mu m$]')
    plt.legend()
    return [pressures, z_sim_hist, z_exp_hist]

#%% 
if __name__ == '__main__':
    debug = False
    if debug:
        Logger = logging.getLogger()
        Logger.setLevel(00) # default is Logger.setLevel(40)
    
    Sim.run_simulation(plotframes=0, 
                       plot_whole_bag = False,
                       printframes=5,
                       simulation_function = 'kinetic_damping',
                       both_sides=False)
    PS.plot(colors='strain')
    x, v = PS.x_v_current_3D
    v_mag = np.linalg.norm(v, axis =1)
    z_max = x[:,2].max()
    f_int = [sd.force_value() for sd in PS.springdampers]
    f_int_mag = np.linalg.norm(f_int, axis=1)
    
    
    
    z_exp = (height_bulge-0.3).roots.real
    z_exp = z_exp[z_exp>0]
    print(f'Expected height:\t{z_exp[0]*1e-6:.2e}')
    print(f'Simulated height:\t{z_max:.2e}')
    print(f'Error: \t\t\t\t{(z_max*1e6-z_exp[0])/z_exp[0]*100:.1f}%')

