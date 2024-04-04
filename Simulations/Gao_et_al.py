# -*- coding: utf-8 -*-
#%% Setup
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from src.particleSystem.ParticleSystem import ParticleSystem
from src.Sim.simulations import Simulate_Lightsail
import src.Mesh.mesh_functions as MF
import src.ExternalForces.optical_interpolators.interpolators as interp
from src.ExternalForces.LaserBeam import LaserBeam
from src.ExternalForces.OpticalForceCalculator import OpticalForceCalculator
from src.ExternalForces.OpticalForceCalculator import ParticleOpticalPropertyType

# Setup parameters
params = {
    # model parameters
    "c": 1,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node
    "thickness":100e-9, # [m] thickness of PhC

    # simulation settings
    "dt": 0.1,  # [s]       simulation timestep
    "t_steps": 1000,  # [-]      max number of simulated time steps
    "abs_tol": 1e-20,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": int(1e2),  # [-]       maximum number of iterations]

    # Simulation Steps
    "steps": np.linspace(0.01,0.1, 25),
    "convergence_threshold": 1e-7,
    "min_iterations":10,

    # Mesh_dependent_settings
    "midstrip_width": 1,
    "boundary_margin": 0.175
    }

params['E'] = 100e9
params['G'] = 0
params['E_x'] = params['E']*7/100
params['E_y'] = params['E']*18/100


# Setup mesh
n_segments = 11 # make sure this is uneven so there are no particles on the centerline
length = 1
mesh = MF.mesh_phc_square_cross(length,
                                mesh_edge_length=length/n_segments,
                                params = params,
                                noncompressive=True)
# We have to add some particles to act as a support structure.
stiffness_support = 1e-1 # [n/m*m] line stiffness
k_support = stiffness_support / (length / n_segments)
l_support = length/n_segments/5

simulate_3D = False

for i in range((n_segments+1)**2):
    # calculate coordinates
    xyz = mesh[1][i][0].copy()
    if xyz[1] ==0 and simulate_3D:
        xyz[1]-=l_support
    elif xyz[1] == length and simulate_3D:
        xyz[1]+=l_support
    elif xyz[0] == 0:
        xyz[0]-=l_support
    elif xyz[0] == length:
        xyz[0]+=l_support


    if np.any(xyz != mesh[1][i][0]):
        particle = [xyz, np.zeros(3), params['m_segment'], True]
        link = [i, len(mesh[1]), k_support, 1]
        mesh[1].append(particle)
        mesh[0].append(link)

    xyz = mesh[1][i][0].copy()
    if (np.all(xyz == [0,0,0])
        or np.all(xyz == [0,length,0])
        or np.all(xyz == [length,0,0])
        or np.all(xyz == [length,length,0])) and simulate_3D:

        if xyz[0] ==0:
            xyz[0]-=l_support
        elif xyz[0] == length:
            xyz[0]+=l_support

        if np.any(xyz != mesh[1][i][0]):
            particle = [xyz, np.zeros(3), params['m_segment'], True]
            link = [i, len(mesh[1]), k_support, 1]
            mesh[1].append(particle)
            mesh[0].append(link)

# init particle system
PS = ParticleSystem(*mesh, params, clean_particles=False)
starting_postions = PS.x_v_current_3D[0]
# Setup the optical sytem
I_0 = 100e9 /(10*10)
mu_x = 0.5
mu_y = 0.5
sigma = 1/2
w=2*length
if simulate_3D:
    LB = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2 # gaussian laser
                                             -1/2 *((y-mu_y)/sigma)**2),
                   lambda x,y: np.outer(np.ones(x.shape),[0,1]))
else:
    LB = LaserBeam(lambda x, y: I_0 * np.exp(-2*((x-mu_x)/w)**2), # gaussian laser
                   lambda x,y: np.outer(np.ones(x.shape),[0,1]))
# Import the crystal
fname = interp.PhC_library['Gao']
#fname = interp.PhC_library['dummy']
interp_right_side = interp.create_interpolator(fname,np.pi)
interp_left_side = interp.create_interpolator(fname, 0)


# set the correct boundary conditions and crystals on the particle system
for p in PS.particles:
    if simulate_3D:
        if p.x[1] == 0 or p.x[1] == length:
            p.set_fixed(True, [0,0,1], 'plane')

    if p.x[0] == 0 or p.x[0] == length:
        p.set_fixed(True, [0,0,1], 'plane')

    p.optical_type = ParticleOpticalPropertyType.ARBITRARY_PHC
    if p.x[0]>length/2:
        p.optical_interpolator = interp_right_side
    else:
        p.optical_interpolator = interp_left_side

OFC = OpticalForceCalculator(PS, LB)
SIM = Simulate_Lightsail(PS,OFC,params)

#%% Plot displaced PS with distributed and net forces

plot_check = False

if plot_check:
    OFC.displace_particle_system([0,0,0,0,5,0])
    forces = OFC.force_value()

    net_force = np.sum(forces,axis=0)
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax = PS.plot(ax)

    COM = OFC.find_center_of_mass()
    x,_ = PS.x_v_current_3D
    OFC.un_displace_particle_system()

    a_u = forces[:,0]
    a_v = forces[:,1]
    a_w = forces[:,2]

    x,y,z = x[:,0], x[:,1], x[:,2]

    ax.quiver(x,y,z,a_u,a_v,a_w, length = 5)
    ax.quiver(COM[0],COM[1],COM[2],
              net_force[0],net_force[1],net_force[2],
              length = 1, label ='Net Force', color='r')

#%% Reproducing Fig. 4 from Gao et al 2022

translations = np.linspace(-length,length,5+2*8*5)
rotations = np.linspace(-10,10,5+2*38)

translation_plot = []
trans_plot = False
if trans_plot:
    fig0 = plt.figure(figsize = [20,16])

resimulate_on_displacement = True

for i, t in enumerate(translations):
    OFC.displace_particle_system([t,0,0,0,0,0])

    if resimulate_on_displacement:
        SIM.run_simulation(plotframes=0, printframes=50, simulation_function='kinetic_damping',file_id=f'_{t}_')
    net_force, net_moments = OFC.calculate_restoring_forces()

    if trans_plot:
        fig0.clear()
        ax0 = fig0.add_subplot(projection='3d')
        ax0.set_xlim([-0.5,1.5])
        ax0.set_ylim([0,1])
        ax0.set_zlim([0,0.5])
        ax0.set_aspect('equal')
        ax0 = PS.plot_forces(OFC.force_value(),ax0)
        ax0.set_title(f'{t:.1f}')
        COM = OFC.find_center_of_mass()
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_force[0],net_force[1],net_force[2],
                  length = 1/2, label ='Net Force', color='r')
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_moments[0],net_moments[1],net_moments[2],
                  length = 2, label ='Net Moment', color='magenta')
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_moments[0],net_moments[1],net_moments[2],
                  length = 1.4, color='magenta')
        fig0.tight_layout()
        fig0.savefig(f'temp/translation-{i}-{t:.2f}.jpg', dpi = 200, format = 'jpg')
    OFC.un_displace_particle_system()

    translation_plot.append([t, *net_force, *net_moments])

rotation_plot=[]
rot_plot = False
if rot_plot:
    fig0 = plt.figure(figsize = [20,16])

for i, r in enumerate(rotations):
    OFC.displace_particle_system([0,0,0,0,r,0])

    if resimulate_on_displacement:
        SIM.run_simulation(plotframes=0, printframes=50, simulation_function='kinetic_damping', file_id=f'_{r}_')
    net_force, net_moments = OFC.calculate_restoring_forces()

    if rot_plot:
        fig0.clear()
        ax0 = fig0.add_subplot(projection='3d')
        ax0 = PS.plot_forces(OFC.force_value(),ax0)
        ax0.set_title(f'{r}')
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_force[0],net_force[1],net_force[2],
                  length = 0.5, label ='Net Force', color='r')
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_moments[0],net_moments[1],net_moments[2],
                  length = 5, label ='Net Moment', color='magenta')
        ax0.quiver(COM[0],COM[1],COM[2],
                  net_moments[0],net_moments[1],net_moments[2],
                  length = 3.5, color='magenta')
        ax0.legend()
        ax0.set_xlim([0,1])
        ax0.set_ylim([0,1])
        ax0.set_zlim([-0.1,0.1])
        ax0.set_aspect('equal')
        fig0.tight_layout()
        fig0.savefig(f'temp/rotation-{i}-{r:.1f}.jpg', dpi = 200, format = 'jpg')

    OFC.un_displace_particle_system()

    rotation_plot.append([r, *net_force, *net_moments])

translation_plot= np.array(translation_plot)
rotation_plot = np.array(rotation_plot)

translation_plot.tofile(f"translation_{stiffness_support=}.csv", sep = ',')
rotation_plot.tofile(f"rotation_{stiffness_support=}.csv", sep = ',')

gao_et_al_figure_four = True
if gao_et_al_figure_four:
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(rotation_plot[:,0], rotation_plot[:,3]/(I_0/c))
    ax1.set_title('Tilt angle versus vertical force')
    ax1.set_xlabel('Tilt angle [deg]')
    ax1.set_ylabel("$F_z [I_0D/c]$")
    ax1.set_ylim([0,rotation_plot[:,3].max()/(I_0/c)*1.2])
    ax1.set_xlim([-10,10])
    ax1.grid()

    ax2 = fig.add_subplot(222)
    ax2.plot(rotation_plot[:,0], rotation_plot[:,5]/(I_0/c))
    ax2.set_title('Tilt angle versus torque')
    ax2.set_xlabel('Tilt angle [deg]')
    ax2.set_ylabel("$\tau_y [I_0D^2/c]$")
    ax2.set_xlim([-10,10])
    ax2.grid()

    ax3 = fig.add_subplot(223)
    ax3.plot(rotation_plot[:,0], -rotation_plot[:,1]/(I_0/c))
    ax3.set_title('Tilt angle versus lateral force')
    ax3.set_xlabel('Tilt angle [deg]')
    ax3.set_ylabel("$F_x [I_0D/c]$")
    ax3.set_xlim([-10,10])
    ax3.grid()

    ax4 = fig.add_subplot(224)
    ax4.plot(translation_plot[:,0], translation_plot[:,1]/(I_0/c))
    ax4.set_title('Translation versus lateral force')
    ax4.set_xlabel('Translation [D]')
    ax4.set_ylabel("$F_x [I_0D/c]$")
    ax4.set_xlim([-1,1])
    ax4.grid()

    fig.tight_layout()


# %% Let's run some simulations!
simulate = False
if simulate:
    SIM.run_simulation(plotframes=0, printframes=10, simulation_function='kinetic_damping', plot_forces=True)
    PS.plot()


