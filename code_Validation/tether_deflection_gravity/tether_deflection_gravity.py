"""
Script for PS framework validation, benchmark case where tether is fixed at both ends and is deflected by perpendicular
gravity which results in a catenary line
"""
import numpy as np
import numpy.typing as npt
import tether_deflection_gravity_input as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


from sympy import *

def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def generate_animation(pos, n: int, t: npt.ArrayLike):
        from matplotlib import animation
        import matplotlib
        import math
        matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\FFmpeg\\bin\\ffmpeg.exe'
        filename = f"Gravity_deflection-{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}" \
                   f"damping_coefficient-{input.params['dt']}timestep-{input.params['t_steps']}steps-.mov"
        savelocation = r"C:\\Users\\Alexander\\Documents\\Master\\Thesis\\Figures\\GIFs\\"

        # configuration of plot
        fig, ax = plt.subplots()
        ax.set_xlim((-1, 11))
        ax.set_ylim((-1, 0.5))
        ax.grid(which='major')
        plt.ylabel("height [m]")
        plt.xlabel("x position [m]")
        plt.title(f"Animation of tether deflected by gravity")

        # calculation which values for each frame
        fps = 60  # 1 / input.params['dt']
        multi = round(input.params['dt']**-1 / fps)
        n_frames = math.floor(len(t)/multi)
        frame_indeces = [i * multi for i in range(n_frames)]

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            index = frame_indeces[i]
            timestep = t[index]
            x = pos.loc[timestep, [f'x{j + 1}' for j in range(n)]]
            y = pos.loc[timestep, [f'z{j + 1}' for j in range(n)]]
            line.set_data(x, y)
            return (line,)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=20, blit=True)  # , save_count=len(self.t))

        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(savelocation + filename, writer=writervideo)
        plt.cla()
        return


def analytical_solution(sag: float, t_l: float):
    # catenary line equation analytical solution
    L = input.params['L']
    h = abs(sag)
    a = (0.25 * t_l**2 - h**2)/(2*h)
    x = np.linspace(0, L, 1000)
    y = a*np.cosh((x-0.5*L)/a)      # shift curve 0.5*L in direction of positive x-axis
    y -= y[0]                       # adjust height
    return x, y


def static_solution(ps: ParticleSystem):
    # hardcoded for 5 particles for now
    import sympy as sp

    k = input.params["k"]       #
    n = input.params["n"]
    l0 = input.params["l0"]
    l = input.params["L"]
    particles = ps.particles
    ux1, uz1 = sp.symbols("ux1 uz1")
    ux2, uz2 = sp.symbols("ux2 uz2")
    ux3, uz3 = sp.symbols("ux3 uz3")
    ux4, uz4 = sp.symbols("ux4 uz4")
    ux5, uz5 = sp.symbols("ux5 uz5")

    F = sp.Matrix([0, -g*particles[0].m, 0, -g*particles[1].m, 0, -g*particles[2].m, 0, -g*particles[3].m,
                   0, -g*particles[4].m])

    U = sp.Matrix([ux1, uz1, ux2, uz2, uz3, uz3, uz4, uz4, uz5, uz5])

    t1 = sp.atan((1 + uz2) / ux2)
    t2 = pi - sp.atan((1 - uy2) / ux2)

    # t1, t2 = sp.symbols("t1 t2")

    K1 = k*sp.Matrix([[sp.cos(t1)*sp.cos(t1), sp.sin(t1)*sp.cos(t1), -sp.cos(t1)*sp.cos(t1), -sp.sin(t1)*sp.cos(t1)],
                      [sp.sin(t1)*sp.cos(t1), sp.sin(t1)*sp.sin(t1), -sp.sin(t1)*sp.cos(t1), -sp.sin(t1)*sp.sin(t1)],
                      [-sp.cos(t1)*sp.cos(t1), -sp.sin(t1)*sp.cos(t1), sp.cos(t1)*sp.cos(t1), sp.sin(t1)*sp.cos(t1)],
                      [-sp.sin(t1)*sp.cos(t1), -sp.sin(t1)*sp.sin(t1), sp.sin(t1)*sp.cos(t1), sp.sin(t1)*sp.sin(t1)]])

    K2 = k*sp.Matrix([[sp.cos(t2)*sp.cos(t2), sp.sin(t2)*sp.cos(t2), -sp.cos(t2)*sp.cos(t2), -sp.sin(t2)*sp.cos(t2)],
                      [sp.sin(t2)*sp.cos(t2), sp.sin(t2)*sp.sin(t2), -sp.sin(t2)*sp.cos(t2), -sp.sin(t2)*sp.sin(t2)],
                      [-sp.cos(t2)*sp.cos(t2), -sp.sin(t2)*sp.cos(t2), sp.cos(t2)*sp.cos(t2), sp.sin(t2)*sp.cos(t2)],
                      [-sp.sin(t2)*sp.cos(t2), -sp.sin(t2)*sp.sin(t2), sp.sin(t2)*sp.cos(t2), sp.sin(t2)*sp.sin(t2)]])

    K = sp.Matrix(zeros(n*2, n*2))      # 2d evaluation for now
    K[0:4, 0:4] += K1
    K[2:, 2:] += K2

    # K = K.col_insert(6, F)
    # print(K)
    U = sp.Matrix([ux2, uy2])
    F = sp.Matrix([f, 0])
    K = K[2:4, 2:4]
    soe = K*U - F
    x = (0, 0, 0.1, 0, 0, 0)

    u = (ux1, uz1, ux2, uz2, ux3, uz3, uz4, uz4, uz5, uz5)
    soe = (soe[0], soe[1], ux1, uz1, ux5, uz5)
    # print(soe)
    # return sp.solve_linear_system(K, U)#  sp.solve(soe, (ux2, uy2))
    solution = sp.solvers.solvers.nsolve(soe, u, x)
    return sp.solvers.solvers.nsolve(soe, u, x)


def plot(psystem: ParticleSystem):
    n = input.params['n']
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])

    x = {}
    v = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))
        v[f"vx{i + 1}"] = np.zeros(len(t_vector))
        v[f"vy{i + 1}"] = np.zeros(len(t_vector))
        v[f"vz{i + 1}"] = np.zeros(len(t_vector))

    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    g = input.params["g"]
    n = input.params["n"]
    f_ext = np.array([[0, 0, -g] for i in range(n)]).flatten()
    print(f_ext)
    particles = ps.particles
    f_ext = np.array([[0, 0, -g*particle.m] for particle in particles]).flatten()
    print(f_ext)
    for step in t_vector:           # propagating the simulation for each timestep and saving results
        position.loc[step], velocity.loc[step] = psystem.simulate(f_ext)

    # generate animation of results, requires smarter configuration to make usable on other PCs
    # generate_animation(position, n, t_vector)

    # generating analytical solution for the same time vector
    tether_length = 0
    particles = ps.particles
    for i in range(n - 1):
        tether_length += np.linalg.norm(particles[i].x - particles[i + 1].x)
    x, y = analytical_solution(min(position.iloc[-1]), tether_length)

    # plotting & graph configuration
    plt.figure(0)
    for i in range(n):
        position[f"z{i + 1}"].plot()
    # plt.plot(t, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Validation PS framework, deflection of particles by gravity, with Implicit Euler scheme")
    plt.legend([f"displacement particle {i + 1}" for i in range(n)])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/" \
                              "tether_deflection_gravity/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}damping_coefficient-" \
               f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    # plot of end state simulation vs analytical solution
    plt.figure(1)
    simx = []
    simy = []
    for i in range(n):
        simx.append(position[f"x{i + 1}"].iloc[-1])
        simy.append(position[f"z{i + 1}"].iloc[-1])

    plt.plot(simx, simy, lw=2)
    plt.plot(x, y)
    plt.grid()
    plt.legend(["Simulation final state particles", "Calculated catenary line"])
    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
    import scipy.sparse.linalg as spla

    jacobian = ps.stiffness_m
    g = input.params['g']
    n = input.params['n']
    b = np.array([[0, 0, -g] for i in range(n)]).flatten()
    x = spla.bicgstab(jacobian, b)

    print(x)
