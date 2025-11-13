mod particle_system;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use particle_system::ParticleSystem as RustParticleSystem;

/// Python wrapper for the Rust ParticleSystem
#[pyclass]
struct ParticleSystemRust {
    inner: RustParticleSystem,
}

#[pymethods]
impl ParticleSystemRust {
    /// Create a new ParticleSystem from Python
    ///
    /// Args:
    ///     connectivity_matrix: List of lists, each containing [i, j, k, c, optional:linktype]
    ///     initial_conditions: List of lists, each containing [x, y, z, vx, vy, vz, mass, fixed, ...]
    ///     sim_params: Dictionary with simulation parameters (dt, rel_tol, abs_tol, max_iter)
    #[new]
    fn new(
        connectivity_matrix: Vec<Vec<f64>>,
        initial_conditions: Vec<Vec<f64>>,
        sim_params: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // Convert Python dict to HashMap
        let mut params = HashMap::new();
        for (key, value) in sim_params.iter() {
            let key_str: String = key.extract()?;
            let val: f64 = value.extract()?;
            params.insert(key_str, val);
        }

        let inner = RustParticleSystem::new(connectivity_matrix, initial_conditions, params);

        Ok(ParticleSystemRust { inner })
    }

    /// Calculate internal forces from spring-dampers
    ///
    /// Returns:
    ///     numpy array of shape (n_particles * 3,) with forces
    fn calculate_forces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let forces = self.inner.calculate_forces();
        forces.into_pyarray(py)
    }

    /// Assemble Jacobian matrices (stiffness and damping)
    ///
    /// Returns:
    ///     Tuple of (jx_data, jx_indices, jx_indptr, jv_data, jv_indices, jv_indptr)
    ///     for creating scipy.sparse.csr_matrix in Python
    fn assemble_jacobians<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
    ) {
        let (jx, jv) = self.inner.assemble_jacobians();

        // Convert to CSR format components for scipy
        let jx_data: Vec<f64> = jx.data().to_vec();
        let jx_indices: Vec<usize> = jx.indices().to_vec();
        let jx_indptr: Vec<usize> = jx.indptr().raw_storage().to_vec();

        let jv_data: Vec<f64> = jv.data().to_vec();
        let jv_indices: Vec<usize> = jv.indices().to_vec();
        let jv_indptr: Vec<usize> = jv.indptr().raw_storage().to_vec();

        (
            jx_data.into_pyarray(py),
            jx_indices.into_pyarray(py),
            jx_indptr.into_pyarray(py),
            jv_data.into_pyarray(py),
            jv_indices.into_pyarray(py),
            jv_indptr.into_pyarray(py),
        )
    }

    /// Get mass matrix in CSR format
    ///
    /// Returns:
    ///     Tuple of (data, indices, indptr) for creating scipy.sparse.csr_matrix
    fn get_mass_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
    ) {
        let m = self.inner.mass_matrix();

        let data: Vec<f64> = m.data().to_vec();
        let indices: Vec<usize> = m.indices().to_vec();
        let indptr: Vec<usize> = m.indptr().raw_storage().to_vec();

        (
            data.into_pyarray(py),
            indices.into_pyarray(py),
            indptr.into_pyarray(py),
        )
    }

    /// Assemble system matrix A and RHS vector b for implicit Euler
    ///
    /// Args:
    ///     f_external: External forces (numpy array of length n_particles * 3)
    ///
    /// Returns:
    ///     Tuple of (A_data, A_indices, A_indptr, b, mask) where:
    ///         - A is system matrix in CSR format: M - dt*Jv - dt²*Jx
    ///         - b is RHS vector: dt*f + dt²*Jx*v
    ///         - mask is boolean array for constraint filtering
    fn assemble_system<'py>(
        &self,
        py: Python<'py>,
        f_external: PyReadonlyArray1<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<f64>>,
        Vec<bool>,
    )> {
        use ndarray::Array1;

        let f_ext = f_external.as_slice()?;
        let f_int = self.inner.calculate_forces();

        // Total force
        let f: Vec<f64> = f_int.iter().zip(f_ext.iter()).map(|(fi, fe)| fi + fe).collect();

        // Get jacobians
        let (jx, jv) = self.inner.assemble_jacobians();

        // Get current state
        let v_current = self.inner.get_velocities();

        // Build system matrix: A = M - dt*Jv - dt²*Jx
        let dt = self.inner.dt;
        let m = self.inner.mass_matrix();

        // Compute: A = M - dt*Jv - dt²*Jx
        // In sparse matrix arithmetic
        let dt2 = dt * dt;

        // Scale jacobians
        let jv_scaled = &jv * (-dt);
        let jx_scaled = &jx * (-dt2);

        // Add matrices - need to clone and add explicitly
        let a = &(m + &jv_scaled) + &jx_scaled;

        // Compute RHS: b = dt*f + dt²*Jx*v
        let mut b = vec![0.0; f.len()];
        for i in 0..f.len() {
            b[i] = dt * f[i];
        }

        // Add dt²*Jx*v to b using manual sparse matrix-vector multiplication
        for row in 0..jx.rows() {
            let mut sum = 0.0;
            for (col, &val) in jx.outer_view(row).unwrap().iter() {
                sum += val * v_current[col];
            }
            b[row] += dt2 * sum;
        }

        // Get constraint mask
        let mask = self.inner.create_constraint_mask();

        // Convert A to CSR format
        let a_data: Vec<f64> = a.data().to_vec();
        let a_indices: Vec<usize> = a.indices().to_vec();
        let a_indptr: Vec<usize> = a.indptr().raw_storage().to_vec();

        Ok((
            a_data.into_pyarray(py),
            a_indices.into_pyarray(py),
            a_indptr.into_pyarray(py),
            b.into_pyarray(py),
            mask,
        ))
    }

    /// Update particle positions and velocities
    ///
    /// Args:
    ///     positions: numpy array of shape (n_particles * 3,)
    ///     velocities: numpy array of shape (n_particles * 3,)
    fn update_state(&mut self, positions: PyReadonlyArray1<f64>, velocities: PyReadonlyArray1<f64>) -> PyResult<()> {
        let pos = positions.as_slice()?.to_vec();
        let vel = velocities.as_slice()?.to_vec();

        self.inner.update_state(pos, vel);
        Ok(())
    }

    /// Get current positions
    ///
    /// Returns:
    ///     numpy array of shape (n_particles * 3,)
    fn get_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.get_positions().into_pyarray(py)
    }

    /// Get current velocities
    ///
    ///Returns:
    ///     numpy array of shape (n_particles * 3,)
    fn get_velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.get_velocities().into_pyarray(py)
    }

    /// Get number of particles
    #[getter]
    fn n_particles(&self) -> usize {
        self.inner.n_particles
    }

    /// Get number of edges (spring-dampers)
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges
    }

    /// Get time step
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt
    }
}

/// Python module for particle_system_rs
#[pymodule]
fn particle_system_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParticleSystemRust>()?;
    Ok(())
}
