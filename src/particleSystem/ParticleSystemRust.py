"""
High-performance Rust-based ParticleSystem implementation with Python interface.

This module provides a drop-in replacement for the Python ParticleSystem class,
using a Rust backend for improved performance through:
- Struct of Arrays (SoA) memory layout for better cache locality
- Sparse matrix representation for efficient memory usage
- Parallel computation using Rayon
- Native compiled code for computational hotspots
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab, cg
import particle_system_rs


class ParticleSystemRust:
    """
    High-performance particle system implementation using Rust backend.

    This class provides the same interface as the Python ParticleSystem but uses
    Rust for core computations. The Rust implementation uses:
    - Sparse CSR matrices for stiffness and damping
    - Struct of Arrays layout for particle data
    - Parallel force and jacobian assembly

    Parameters
    ----------
    connectivity_matrix : list
        List of connections, each containing [i, j, k, c, optional:linktype]
    initial_conditions : npt.ArrayLike
        Array of initial conditions for each particle:
        [x, y, z, vx, vy, vz, mass, fixed, constraint_x, constraint_y, constraint_z, constraint_type]
    sim_param : dict
        Simulation parameters (dt, rel_tol, abs_tol, max_iter)
    use_cg_solver : bool, optional
        If True, use Conjugate Gradient solver instead of BiCGSTAB (default: True)
        CG is faster for SPD systems
    """

    def __init__(self,
                 connectivity_matrix: list,
                 initial_conditions: npt.ArrayLike,
                 sim_param: dict,
                 use_cg_solver: bool = True,
                 **kwargs):

        self.__params = sim_param
        self.__use_cg = use_cg_solver

        # Convert connectivity matrix - handle SpringDamperType enums
        try:
            from ..particleSystem.SpringDamper import SpringDamperType
        except ImportError:
            from particleSystem.SpringDamper import SpringDamperType
        rust_connectivity = []
        for conn in connectivity_matrix:
            conn_copy = [float(conn[0]), float(conn[1]), float(conn[2]), float(conn[3])]  # [i, j, k, c]
            if len(conn) >= 5:
                # Convert SpringDamperType enum to int
                link_type = conn[4]
                if hasattr(link_type, 'value'):
                    # Handle enum by checking value
                    val_str = str(link_type.value).lower()
                    if 'noncompressive' in val_str:
                        conn_copy.append(1.0)
                    elif 'nontensile' in val_str:
                        conn_copy.append(2.0)
                    else:
                        conn_copy.append(0.0)
                elif isinstance(link_type, (int, float)):
                    conn_copy.append(float(link_type))
                else:
                    # Default to 0
                    conn_copy.append(0.0)
            rust_connectivity.append(conn_copy)

        # Convert initial conditions to the format expected by Rust
        # Python format: [[pos, vel, mass, fixed, constraint, constraint_type], ...]
        # Rust format: [[x, y, z, vx, vy, vz, mass, fixed, cx, cy, cz, ctype], ...]
        rust_initial_conditions = []
        for ic in initial_conditions:
            pos = np.asarray(ic[0])  # [x, y, z]
            vel = np.asarray(ic[1])  # [vx, vy, vz]
            mass = float(ic[2])
            fixed = 1.0 if ic[3] else 0.0

            # Handle constraints
            if fixed and len(ic) >= 5:
                constraint = np.asarray(ic[4])  # Constraint vector
                constraint_type = ic[5] if len(ic) >= 6 else 'point'

                # Map constraint type to number
                ctype_map = {'free': 0, 'point': 1, 'line': 2, 'plane': 3}
                ctype = ctype_map.get(constraint_type, 1)

                rust_ic = [float(pos[0]), float(pos[1]), float(pos[2]),
                          float(vel[0]), float(vel[1]), float(vel[2]),
                          mass, fixed,
                          float(constraint[0]), float(constraint[1]), float(constraint[2]), float(ctype)]
            else:
                rust_ic = [float(pos[0]), float(pos[1]), float(pos[2]),
                          float(vel[0]), float(vel[1]), float(vel[2]),
                          mass, fixed, 0.0, 0.0, 0.0, 1.0]

            rust_initial_conditions.append(rust_ic)

        # Create Rust particle system
        self.__ps_rust = particle_system_rs.ParticleSystemRust(
            rust_connectivity,
            rust_initial_conditions,
            sim_param
        )

        self.__n = self.__ps_rust.n_particles
        self.__dt = sim_param["dt"]
        self.__rtol = sim_param["rel_tol"]
        self.__atol = sim_param["abs_tol"]
        self.__maxiter = int(sim_param["max_iter"])

        # For compatibility with existing code
        self.__history = {'dt': [], 'E_kin': []}

    def simulate(self, f_external: npt.ArrayLike = ()):
        """
        Advance simulation by one timestep using implicit Euler method.

        This method assembles the system matrix in Rust (using sparse CSR format),
        then solves the linear system using scipy's iterative solvers.

        Parameters
        ----------
        f_external : npt.ArrayLike, optional
            External forces (length = n_particles * 3)

        Returns
        -------
        x_next : np.ndarray
            Updated positions
        v_next : np.ndarray
            Updated velocities
        """
        if not len(f_external):
            f_external = np.zeros(self.__n * 3, dtype=np.float64)

        # Assemble system in Rust: A*dv = b
        # Returns A as CSR components and b as vector
        A_data, A_indices, A_indptr, b, mask = self.__ps_rust.assemble_system(f_external)

        # Create scipy sparse matrix
        shape = (self.__n * 3, self.__n * 3)
        A = csr_matrix((A_data, A_indices, A_indptr), shape=shape)

        # Filter constrained DOFs
        mask = np.array(mask, dtype=bool)
        A_filtered = A[mask, :][:, mask]
        b_filtered = b[mask]

        # Solve linear system
        if self.__use_cg:
            # Use Conjugate Gradient for SPD systems (typically faster)
            dv_filtered, info = cg(A_filtered, b_filtered,
                                   rtol=self.__rtol, atol=self.__atol,
                                   maxiter=self.__maxiter)
        else:
            # Use BiCGSTAB (original solver)
            dv_filtered, info = bicgstab(A_filtered, b_filtered,
                                         rtol=self.__rtol, atol=self.__atol,
                                         maxiter=self.__maxiter)

        # Map solution back to full vector
        dv = np.zeros(self.__n * 3, dtype=np.float64)
        dv[mask] = dv_filtered

        # Get current state
        v_current = self.__ps_rust.get_velocities()
        x_current = self.__ps_rust.get_positions()

        # Time integration
        v_next = v_current + dv
        x_next = x_current + self.__dt * v_next

        # Update state in Rust
        self.__ps_rust.update_state(x_next, v_next)

        # Record history
        self.__history['dt'].append(self.__dt)
        # Note: E_kin calculation would require accessing particle masses

        return x_next, v_next

    def calculate_forces(self):
        """
        Calculate internal forces from all spring-dampers.

        Returns
        -------
        forces : np.ndarray
            Force vector (length = n_particles * 3)
        """
        return self.__ps_rust.calculate_forces()

    def get_positions(self):
        """Get current particle positions."""
        return self.__ps_rust.get_positions()

    def get_velocities(self):
        """Get current particle velocities."""
        return self.__ps_rust.get_velocities()

    def update_state(self, positions: npt.ArrayLike, velocities: npt.ArrayLike):
        """Update particle positions and velocities."""
        self.__ps_rust.update_state(np.asarray(positions), np.asarray(velocities))

    @property
    def x_v_current(self):
        """Get current positions and velocities as flattened arrays."""
        return self.__ps_rust.get_positions(), self.__ps_rust.get_velocities()

    @property
    def x_v_current_3D(self):
        """Get current positions and velocities as (n, 3) arrays."""
        x = self.__ps_rust.get_positions()
        v = self.__ps_rust.get_velocities()
        x_3d = x.reshape((self.__n, 3))
        v_3d = v.reshape((self.__n, 3))
        return x_3d, v_3d

    @property
    def n(self):
        """Number of particles."""
        return self.__n

    @property
    def params(self):
        """Simulation parameters."""
        return self.__params

    @property
    def history(self):
        """Simulation history."""
        return self.__history

    def reset_history(self):
        """Reset simulation history."""
        self.__history = {'dt': [], 'E_kin': []}


if __name__ == "__main__":
    # Simple test
    params = {
        "dt": 0.001,
        "rel_tol": 1e-5,
        "abs_tol": 1e-50,
        "max_iter": int(1e5),
    }

    # Create a simple 3-particle system
    c_matrix = [[0, 1, 2e4, 0.0],
                [1, 2, 2e4, 0.0]]

    init_cond = [
        [[0, 0, 0], [0, 0, 0], 1.0, True],
        [[1, 0, 0], [0, 0, 0], 1.0, False],
        [[1, 1, 0], [0, 0, 0], 1.0, False]
    ]

    ps = ParticleSystemRust(c_matrix, init_cond, params)
    print(f"Created ParticleSystem with {ps.n} particles")

    # Run a few timesteps
    for i in range(10):
        x, v = ps.simulate()

    print(f"Final positions:\n{ps.x_v_current_3D[0]}")
    print("Test completed successfully!")
