"""
Benchmark and test script for comparing Python and Rust ParticleSystem implementations.

This script:
1. Creates identical particle systems in Python and Rust
2. Runs simulations and compares results for correctness
3. Benchmarks performance of both implementations
"""

import time
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from particleSystem.ParticleSystem import ParticleSystem
from particleSystem.ParticleSystemRust import ParticleSystemRust


def create_mesh_grid(nx, ny, spacing=0.1, stiffness=1e4, damping=10.0):
    """
    Create a simple 2D mesh grid for testing.

    Parameters
    ----------
    nx, ny : int
        Number of nodes in x and y directions
    spacing : float
        Distance between nodes
    stiffness : float
        Spring stiffness
    damping : float
        Damping coefficient

    Returns
    -------
    connectivity_matrix : list
        Connectivity matrix
    initial_conditions : list
        Initial conditions
    """
    n_particles = nx * ny

    # Create initial conditions
    initial_conditions = []
    for j in range(ny):
        for i in range(nx):
            x = i * spacing
            y = j * spacing
            z = 0.0

            # Fix bottom edge
            fixed = (j == 0)

            initial_conditions.append([
                [x, y, z],  # position
                [0.0, 0.0, 0.0],  # velocity
                1.0,  # mass
                fixed  # fixed
            ])

    # Create connectivity (connect each node to its neighbors)
    connectivity_matrix = []

    def get_index(i, j):
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            idx = get_index(i, j)

            # Connect to right neighbor
            if i < nx - 1:
                connectivity_matrix.append([idx, get_index(i + 1, j), stiffness, damping])

            # Connect to top neighbor
            if j < ny - 1:
                connectivity_matrix.append([idx, get_index(i, j + 1), stiffness, damping])

            # Connect to diagonal neighbor (top-right)
            if i < nx - 1 and j < ny - 1:
                connectivity_matrix.append([idx, get_index(i + 1, j + 1), stiffness, damping])

            # Connect to diagonal neighbor (top-left)
            if i > 0 and j < ny - 1:
                connectivity_matrix.append([idx, get_index(i - 1, j + 1), stiffness, damping])

    return connectivity_matrix, initial_conditions


def test_correctness(n_steps=10, mesh_size=(5, 5)):
    """
    Test that Python and Rust implementations produce similar results.
    """
    print("=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)

    nx, ny = mesh_size
    print(f"Creating {nx}x{ny} mesh grid ({nx*ny} particles)...")

    connectivity_matrix, initial_conditions = create_mesh_grid(nx, ny)

    params = {
        "dt": 0.001,
        "rel_tol": 1e-5,
        "abs_tol": 1e-50,
        "max_iter": int(1e5),
    }

    print(f"Number of particles: {len(initial_conditions)}")
    print(f"Number of springs: {len(connectivity_matrix)}")

    # Create both implementations
    print("\nCreating Python ParticleSystem...")
    ps_python = ParticleSystem(
        connectivity_matrix.copy(),
        [ic.copy() for ic in initial_conditions],
        params
    )

    print("Creating Rust ParticleSystem...")
    ps_rust = ParticleSystemRust(
        connectivity_matrix.copy(),
        [ic.copy() for ic in initial_conditions],
        params
    )

    # Apply external force (gravity)
    g = 9.81
    n_particles = len(initial_conditions)
    f_ext = np.zeros(n_particles * 3)
    f_ext[2::3] = -g  # Apply gravity in -z direction

    print(f"\nRunning {n_steps} simulation steps...")

    # Run simulations
    for i in range(n_steps):
        x_py, v_py = ps_python.simulate(f_ext.copy())
        x_rs, v_rs = ps_rust.simulate(f_ext.copy())

    # Compare results
    print("\n" + "-" * 70)
    print("RESULTS COMPARISON")
    print("-" * 70)

    max_pos_diff = np.max(np.abs(x_py - x_rs))
    max_vel_diff = np.max(np.abs(v_py - v_rs))
    mean_pos_diff = np.mean(np.abs(x_py - x_rs))
    mean_vel_diff = np.mean(np.abs(v_py - v_rs))

    print(f"Position difference:")
    print(f"  Max:  {max_pos_diff:.3e}")
    print(f"  Mean: {mean_pos_diff:.3e}")
    print(f"\nVelocity difference:")
    print(f"  Max:  {max_vel_diff:.3e}")
    print(f"  Mean: {mean_vel_diff:.3e}")

    # Check if differences are acceptable
    # Note: Small numerical differences are expected between implementations
    tolerance = 1e-5
    if max_pos_diff < tolerance and max_vel_diff < tolerance:
        print(f"\n✓ PASSED: Differences are within tolerance ({tolerance:.1e})")
        return True
    else:
        print(f"\n✗ FAILED: Differences exceed tolerance ({tolerance:.1e})")
        return False


def benchmark_performance(n_steps=100, mesh_sizes=[(5, 5), (10, 10), (20, 20)]):
    """
    Benchmark performance of Python vs Rust implementations.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    results = []

    for nx, ny in mesh_sizes:
        print(f"\n{'─' * 70}")
        print(f"Mesh size: {nx}x{ny} ({nx*ny} particles)")
        print(f"{'─' * 70}")

        connectivity_matrix, initial_conditions = create_mesh_grid(nx, ny)

        params = {
            "dt": 0.001,
            "rel_tol": 1e-5,
            "abs_tol": 1e-50,
            "max_iter": int(1e5),
        }

        n_particles = len(initial_conditions)
        f_ext = np.zeros(n_particles * 3)
        f_ext[2::3] = -9.81

        # Benchmark Python
        print(f"Benchmarking Python implementation...")
        ps_python = ParticleSystem(
            connectivity_matrix.copy(),
            [ic.copy() for ic in initial_conditions],
            params
        )

        start = time.time()
        for i in range(n_steps):
            ps_python.simulate(f_ext.copy())
        time_python = time.time() - start

        print(f"  Time: {time_python:.3f} seconds")
        print(f"  Time per step: {time_python/n_steps*1000:.2f} ms")

        # Benchmark Rust
        print(f"Benchmarking Rust implementation...")
        ps_rust = ParticleSystemRust(
            connectivity_matrix.copy(),
            [ic.copy() for ic in initial_conditions],
            params,
            use_cg_solver=True  # CG is typically faster for SPD systems
        )

        start = time.time()
        for i in range(n_steps):
            ps_rust.simulate(f_ext.copy())
        time_rust = time.time() - start

        print(f"  Time: {time_rust:.3f} seconds")
        print(f"  Time per step: {time_rust/n_steps*1000:.2f} ms")

        speedup = time_python / time_rust
        print(f"\n  Speedup: {speedup:.2f}x")

        results.append({
            'mesh_size': (nx, ny),
            'n_particles': nx * ny,
            'n_springs': len(connectivity_matrix),
            'time_python': time_python,
            'time_rust': time_rust,
            'speedup': speedup
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mesh Size':<12} {'Particles':<12} {'Springs':<12} {'Speedup':<12}")
    print("-" * 70)
    for r in results:
        print(f"{str(r['mesh_size']):<12} {r['n_particles']:<12} {r['n_springs']:<12} {r['speedup']:.2f}x")

    avg_speedup = np.mean([r['speedup'] for r in results])
    print("-" * 70)
    print(f"Average speedup: {avg_speedup:.2f}x")

    return results


def main():
    print("Particle System: Rust vs Python Comparison")
    print("=" * 70)

    # Run correctness test
    success = test_correctness(n_steps=10, mesh_size=(5, 5))

    if not success:
        print("\n⚠ Correctness test failed. Skipping performance benchmark.")
        return

    # Run performance benchmark
    benchmark_performance(n_steps=50, mesh_sizes=[(5, 5), (10, 10), (15, 15)])

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
