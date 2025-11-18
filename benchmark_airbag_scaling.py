"""
Airbag Scaling Benchmark - Push the limits of the Rust ParticleSystem

This script progressively scales up the airbag simulation to find the maximum
size that can be solved within a 30-second time limit. It includes:
- Automatic mesh scaling
- Convergence detection based on velocity magnitude
- 3D visualization of final shapes
- Performance comparison Python vs Rust

Note: Uses regular simulate() with dynamic relaxation instead of kinetic damping.
The approach: run with light damping and monitor for convergence.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from particleSystem.ParticleSystem import ParticleSystem
from particleSystem.ParticleSystemRust import ParticleSystemRust
import src.Mesh.mesh_functions as MF


def create_airbag_mesh(n_segments, half_width=0.422, params=None):
    """
    Create an airbag mesh with given parameters.

    Parameters
    ----------
    n_segments : int
        Number of segments along each edge
    half_width : float
        Half-width of the airbag in meters
    params : dict
        Simulation parameters

    Returns
    -------
    initial_conditions, connections, params : tuple
        Mesh data
    """
    if params is None:
        params = {
            "k": 250,
            "k_d": 250,
            "c": 50,  # Increased damping for dynamic relaxation
            "m_segment": 1,
            "E": 0.588,  # GPa
            "thickness": 0.6e-3,
            "rho": 333,
            "dt": 0.005,  # Smaller timestep for stability
            "t_steps": 1e4,
            "abs_tol": 1e-50,
            "rel_tol": 1e-5,
            "max_iter": int(1e5),
            "convergence_threshold": 1e-6,  # Relaxed for dynamic simulation
            "min_iterations": 10,
            "pressure": 5e3,
        }

    # Calculate spring stiffness based on mesh resolution
    diagonal_spring_ratio = 0.625
    params['k'] = params["E"] * 1e9 * params['thickness'] * n_segments / \
                  (n_segments + 1 + 2 * n_segments * diagonal_spring_ratio / np.sqrt(2))
    params['k_d'] = params['k'] * diagonal_spring_ratio

    edge_length = half_width / n_segments

    initial_conditions, connections = MF.mesh_airbag_square_cross(
        half_width,
        mesh_edge_length=edge_length,
        params=params,
        noncompressive=True
    )

    return initial_conditions, connections, params


def calculate_airbag_forces(PS, pressure, half_width=0.422):
    """Calculate pressure forces on airbag surface.

    For the airbag problem, pressure acts normal to the membrane surface.
    We approximate by applying uniform upward (+Z) force on all particles.
    """
    n_particles = PS.n
    f_ext = np.zeros(n_particles * 3)

    # Find surface areas using Python PS method if available
    try:
        if hasattr(PS, 'find_surface'):
            surface_areas = PS.find_surface()
            f_ext = (surface_areas * pressure).flatten()
            return f_ext
    except Exception as e:
        pass

    # Fallback: uniform pressure approximation
    # Total area of the quarter airbag = half_width^2
    # Distribute pressure force among all particles
    total_area = half_width * half_width
    area_per_particle = total_area / n_particles
    force_per_particle = pressure * area_per_particle

    # Apply force in +Z direction on all particles
    # This is a simplification - real pressure acts normal to surface
    f_ext[2::3] = force_per_particle

    print(f"Applied uniform pressure: {force_per_particle:.2f} N per particle")

    return f_ext


def check_convergence(kinetic_energies, threshold=1e-8, window=5):
    """Check if simulation has converged."""
    if len(kinetic_energies) < window:
        return False

    recent = kinetic_energies[-window:]
    change = abs(recent[-1] - recent[-2])

    return change < threshold


def run_airbag_simulation(n_segments, time_limit=30, use_rust=True, visualize=True):
    """
    Run airbag simulation with given mesh size.

    Parameters
    ----------
    n_segments : int
        Mesh resolution (n_segments x n_segments grid)
    time_limit : float
        Maximum simulation time in seconds
    use_rust : bool
        Use Rust implementation if True
    visualize : bool
        Generate 3D plot of final state

    Returns
    -------
    results : dict
        Simulation results and statistics
    """
    print(f"\n{'='*70}")
    print(f"Testing {n_segments}x{n_segments} airbag mesh")
    print(f"{'='*70}")

    # Create mesh
    initial_conditions, connections, params = create_airbag_mesh(n_segments)
    n_particles = len(initial_conditions)
    n_springs = len(connections)

    print(f"Particles: {n_particles}")
    print(f"Springs: {n_springs}")

    # Create particle system
    if use_rust:
        PS = ParticleSystemRust(connections, initial_conditions, params, use_cg_solver=True)
        impl_name = "Rust"
    else:
        PS = ParticleSystem(connections, initial_conditions, params)
        impl_name = "Python"

    print(f"Using {impl_name} implementation")

    # Calculate pressure forces
    pressure = params['pressure']
    f_ext = calculate_airbag_forces(PS, pressure)

    # Run simulation
    start_time = time.time()
    step = 0
    converged = False
    convergence_history = []

    print(f"\nRunning simulation (time limit: {time_limit}s)...")

    while not converged and (time.time() - start_time) < time_limit:
        # Simulate one step
        try:
            x, v = PS.simulate(f_ext)

            # Track kinetic energy for convergence
            v_norm = np.linalg.norm(v)
            convergence_history.append(v_norm)

            step += 1

            # Check convergence every 10 steps
            if step % 10 == 0:
                elapsed = time.time() - start_time
                if check_convergence(convergence_history, threshold=params['convergence_threshold']):
                    converged = True
                    print(f"✓ Converged at step {step} ({elapsed:.2f}s)")
                elif step % 50 == 0:
                    print(f"Step {step}: v_norm={v_norm:.3e}, elapsed={elapsed:.2f}s")

            # Safety check - if velocities are extremely large, something went wrong
            if v_norm > 1e6:
                print(f"✗ Simulation diverged (v_norm={v_norm:.2e})")
                return None

        except Exception as e:
            print(f"✗ Simulation failed: {e}")
            return None

    elapsed_time = time.time() - start_time

    if not converged:
        print(f"⚠ Time limit reached ({elapsed_time:.2f}s), {step} steps completed")

    # Get final state
    x_final, v_final = PS.x_v_current_3D

    # Calculate statistics
    displacement = np.linalg.norm(x_final - x_final.mean(axis=0), axis=1).max()

    results = {
        'n_segments': n_segments,
        'n_particles': n_particles,
        'n_springs': n_springs,
        'converged': converged,
        'steps': step,
        'time': elapsed_time,
        'time_per_step': elapsed_time / step if step > 0 else 0,
        'max_displacement': displacement,
        'final_v_norm': convergence_history[-1] if convergence_history else 0,
        'x_final': x_final,
        'convergence_history': convergence_history,
        'PS': PS,
    }

    print(f"\nResults:")
    print(f"  Steps: {step}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Time/step: {results['time_per_step']*1000:.2f}ms")
    print(f"  Max displacement: {displacement:.3f}m")

    # Visualization
    if visualize and x_final is not None:
        fig = plt.figure(figsize=(12, 5))

        # 3D view of final shape
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(x_final[:, 0], x_final[:, 1], x_final[:, 2],
                   c=x_final[:, 2], cmap='viridis', s=20)
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title(f'Final Airbag Shape ({n_segments}x{n_segments})')
        ax1.set_box_aspect([1, 1, 0.5])

        # Convergence plot
        ax2 = fig.add_subplot(122)
        ax2.semilogy(convergence_history)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity Norm')
        ax2.set_title('Convergence History')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'airbag_{n_segments}x{n_segments}_{impl_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {filename}")
        plt.close()

    return results


def run_scaling_benchmark(time_limit=30, start_size=10, max_size=100, use_rust=True):
    """
    Run progressive scaling benchmark to find maximum feasible size.

    Parameters
    ----------
    time_limit : float
        Time limit per simulation in seconds
    start_size : int
        Starting mesh size (n x n)
    max_size : int
        Maximum mesh size to try
    use_rust : bool
        Use Rust implementation

    Returns
    -------
    all_results : list
        Results for each tested size
    """
    print("="*70)
    print("AIRBAG SCALING BENCHMARK")
    print("="*70)
    print(f"Time limit per size: {time_limit}s")
    print(f"Implementation: {'Rust' if use_rust else 'Python'}")

    all_results = []

    # Progressive scaling strategy
    # Start with reasonable size, then scale up
    test_sizes = []
    size = start_size
    while size <= max_size:
        test_sizes.append(size)
        # Increase more slowly as we get larger
        if size < 20:
            size += 5
        elif size < 40:
            size += 10
        else:
            size += 20

    for n_segments in test_sizes:
        result = run_airbag_simulation(n_segments, time_limit=time_limit,
                                      use_rust=use_rust, visualize=True)

        if result is None:
            print(f"\n⚠ Skipping larger sizes - {n_segments}x{n_segments} failed")
            break

        all_results.append(result)

        # If we didn't converge or took too long, stop scaling
        if not result['converged']:
            print(f"\n⚠ Stopping - {n_segments}x{n_segments} didn't converge in time limit")
            break

        # If we used >80% of time limit, might want to stop
        if result['time'] > 0.8 * time_limit:
            print(f"\n⚠ Stopping - getting close to time limit")
            break

    # Generate summary
    print("\n" + "="*70)
    print("SCALING BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Size':<12} {'Particles':<12} {'Springs':<12} {'Steps':<12} {'Time':<12} {'ms/step':<12}")
    print("-"*70)

    for r in all_results:
        status = "✓" if r['converged'] else "⚠"
        print(f"{status} {r['n_segments']}x{r['n_segments']:<8} "
              f"{r['n_particles']:<12} {r['n_springs']:<12} "
              f"{r['steps']:<12} {r['time']:<12.2f} {r['time_per_step']*1000:<12.2f}")

    if all_results:
        max_size_converged = max([r['n_segments'] for r in all_results if r['converged']], default=0)
        print("-"*70)
        print(f"Maximum converged size: {max_size_converged}x{max_size_converged}")
        print(f"Maximum particles: {max([r['n_particles'] for r in all_results], default=0)}")

    return all_results


if __name__ == "__main__":
    # Run the scaling benchmark
    results_rust = run_scaling_benchmark(
        time_limit=30,
        start_size=10,
        max_size=100,
        use_rust=True
    )

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results_rust)} visualizations")
    print("Check the current directory for airbag_*.png files")
