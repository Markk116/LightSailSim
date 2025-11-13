/// Core ParticleSystem implementation using Struct of Arrays (SoA) layout
/// for cache-friendly memory access patterns.

use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;

/// Link type enumeration matching Python SpringDamperType
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkType {
    Default = 0,
    NonCompressive = 1,
    NonTensile = 2,
}

impl LinkType {
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => LinkType::NonCompressive,
            2 => LinkType::NonTensile,
            _ => LinkType::Default,
        }
    }
}

/// Constraint type for particles
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    Free,
    Point,
    Line,
    Plane,
}

/// Core ParticleSystem with Struct of Arrays layout
/// This layout improves cache locality for vectorized operations
pub struct ParticleSystem {
    // Particle data (SoA layout) - each Vec contains data for all particles
    pub positions: Vec<f64>,        // [x0, y0, z0, x1, y1, z1, ...]
    pub velocities: Vec<f64>,       // [vx0, vy0, vz0, vx1, vy1, vz1, ...]
    pub masses: Vec<f64>,           // [m0, m1, m2, ...]
    pub fixed: Vec<bool>,           // [f0, f1, f2, ...]
    pub constraint_types: Vec<ConstraintType>,
    pub constraints: Vec<[f64; 3]>, // Constraint vectors

    // Spring-damper connectivity (SoA layout)
    pub edge_i: Vec<usize>,         // First particle indices
    pub edge_j: Vec<usize>,         // Second particle indices
    pub stiffness: Vec<f64>,        // Spring stiffness k
    pub damping: Vec<f64>,          // Damping coefficient c
    pub rest_lengths: Vec<f64>,     // Rest lengths l0
    pub link_types: Vec<LinkType>,  // Link types

    // Simulation parameters
    pub dt: f64,
    pub rtol: f64,
    pub atol: f64,
    pub maxiter: usize,

    // System dimensions
    pub n_particles: usize,
    pub n_edges: usize,

    // Cached matrices (sparse format)
    mass_matrix: Option<CsMat<f64>>,
}

impl ParticleSystem {
    /// Create a new ParticleSystem from connectivity matrix and initial conditions
    pub fn new(
        connectivity_matrix: Vec<Vec<f64>>,
        initial_conditions: Vec<Vec<f64>>,
        sim_params: HashMap<String, f64>,
    ) -> Self {
        let n_particles = initial_conditions.len();
        let n_edges = connectivity_matrix.len();

        // Pre-allocate vectors with correct capacity
        let mut positions = Vec::with_capacity(n_particles * 3);
        let mut velocities = Vec::with_capacity(n_particles * 3);
        let mut masses = Vec::with_capacity(n_particles);
        let mut fixed = Vec::with_capacity(n_particles);
        let mut constraint_types = Vec::with_capacity(n_particles);
        let mut constraints = Vec::with_capacity(n_particles);

        // Parse initial conditions
        for ic in initial_conditions.iter() {
            // Format: [pos[3], vel[3], mass, fixed, constraint[3] (optional), constraint_type (optional)]
            positions.extend_from_slice(&ic[0..3]);
            velocities.extend_from_slice(&ic[3..6]);
            masses.push(ic[6]);
            fixed.push(ic[7] != 0.0);

            if ic.len() >= 11 {
                constraints.push([ic[8], ic[9], ic[10]]);
                let ct = if ic.len() >= 12 {
                    match ic[11] as i32 {
                        1 => ConstraintType::Point,
                        2 => ConstraintType::Line,
                        3 => ConstraintType::Plane,
                        _ => ConstraintType::Free,
                    }
                } else {
                    ConstraintType::Point
                };
                constraint_types.push(ct);
            } else {
                constraints.push([0.0, 0.0, 0.0]);
                constraint_types.push(if ic[7] != 0.0 { ConstraintType::Point } else { ConstraintType::Free });
            }
        }

        // Pre-allocate connectivity vectors
        let mut edge_i = Vec::with_capacity(n_edges);
        let mut edge_j = Vec::with_capacity(n_edges);
        let mut stiffness = Vec::with_capacity(n_edges);
        let mut damping = Vec::with_capacity(n_edges);
        let mut rest_lengths = Vec::with_capacity(n_edges);
        let mut link_types = Vec::with_capacity(n_edges);

        // Parse connectivity matrix
        for conn in connectivity_matrix.iter() {
            // Format: [i, j, k, c, optional: linktype]
            edge_i.push(conn[0] as usize);
            edge_j.push(conn[1] as usize);
            stiffness.push(conn[2]);
            damping.push(conn[3]);

            let link_type = if conn.len() >= 5 {
                LinkType::from_u8(conn[4] as u8)
            } else {
                LinkType::Default
            };
            link_types.push(link_type);
        }

        // Compute rest lengths
        for idx in 0..n_edges {
            let i = edge_i[idx];
            let j = edge_j[idx];
            let dx = positions[3*i] - positions[3*j];
            let dy = positions[3*i + 1] - positions[3*j + 1];
            let dz = positions[3*i + 2] - positions[3*j + 2];
            let l0 = (dx*dx + dy*dy + dz*dz).sqrt();
            rest_lengths.push(l0);
        }

        // Extract simulation parameters
        let dt = sim_params.get("dt").copied().unwrap_or(0.001);
        let rtol = sim_params.get("rel_tol").copied().unwrap_or(1e-5);
        let atol = sim_params.get("abs_tol").copied().unwrap_or(1e-50);
        let maxiter = sim_params.get("max_iter").copied().unwrap_or(1e5) as usize;

        let mut ps = ParticleSystem {
            positions,
            velocities,
            masses,
            fixed,
            constraint_types,
            constraints,
            edge_i,
            edge_j,
            stiffness,
            damping,
            rest_lengths,
            link_types,
            dt,
            rtol,
            atol,
            maxiter,
            n_particles,
            n_edges,
            mass_matrix: None,
        };

        // Build mass matrix
        ps.build_mass_matrix();

        ps
    }

    /// Build sparse mass matrix (diagonal)
    fn build_mass_matrix(&mut self) {
        let n_dof = self.n_particles * 3;
        let mut triplets = TriMat::new((n_dof, n_dof));

        for i in 0..self.n_particles {
            let m = self.masses[i];
            triplets.add_triplet(3*i, 3*i, m);
            triplets.add_triplet(3*i + 1, 3*i + 1, m);
            triplets.add_triplet(3*i + 2, 3*i + 2, m);
        }

        self.mass_matrix = Some(triplets.to_csr());
    }

    /// Calculate internal forces from all spring-dampers
    /// Returns force vector (length = 3 * n_particles)
    pub fn calculate_forces(&self) -> Vec<f64> {
        let n_dof = self.n_particles * 3;
        let mut forces = vec![0.0; n_dof];

        // Parallel force computation using rayon
        let partial_forces: Vec<Vec<f64>> = self.edge_i.par_iter()
            .enumerate()
            .map(|(edge_idx, &i)| {
                let j = self.edge_j[edge_idx];
                let k = self.stiffness[edge_idx];
                let c = self.damping[edge_idx];
                let l0 = self.rest_lengths[edge_idx];
                let link_type = self.link_types[edge_idx];

                // Compute relative position
                let dx = self.positions[3*i] - self.positions[3*j];
                let dy = self.positions[3*i + 1] - self.positions[3*j + 1];
                let dz = self.positions[3*i + 2] - self.positions[3*j + 2];
                let l = (dx*dx + dy*dy + dz*dz).sqrt();

                // Check link type constraints
                let is_active = match link_type {
                    LinkType::NonCompressive => l >= l0,
                    LinkType::NonTensile => l <= l0,
                    LinkType::Default => true,
                };

                if !is_active || l < 1e-10 {
                    return vec![0.0; n_dof];
                }

                // Unit vector
                let ux = dx / l;
                let uy = dy / l;
                let uz = dz / l;

                // Spring force
                let f_spring_mag = -k * (l - l0);

                // Damping force (velocity component along spring direction)
                let dvx = self.velocities[3*i] - self.velocities[3*j];
                let dvy = self.velocities[3*i + 1] - self.velocities[3*j + 1];
                let dvz = self.velocities[3*i + 2] - self.velocities[3*j + 2];
                let v_rel = dvx * ux + dvy * uy + dvz * uz;
                let f_damp_mag = -c * v_rel;

                // Total force magnitude
                let f_mag = f_spring_mag + f_damp_mag;

                // Force components
                let fx = f_mag * ux;
                let fy = f_mag * uy;
                let fz = f_mag * uz;

                // Create local force vector
                let mut local_f = vec![0.0; n_dof];
                local_f[3*i] = fx;
                local_f[3*i + 1] = fy;
                local_f[3*i + 2] = fz;
                local_f[3*j] = -fx;
                local_f[3*j + 1] = -fy;
                local_f[3*j + 2] = -fz;

                local_f
            })
            .collect();

        // Reduce parallel results
        for partial in partial_forces {
            for (i, &f) in partial.iter().enumerate() {
                forces[i] += f;
            }
        }

        forces
    }

    /// Assemble sparse stiffness and damping Jacobians
    /// Returns (jx, jv) as sparse CSR matrices
    pub fn assemble_jacobians(&self) -> (CsMat<f64>, CsMat<f64>) {
        let n_dof = self.n_particles * 3;

        // Estimate number of non-zeros: each edge contributes to 4 3x3 blocks
        let nnz_estimate = self.n_edges * 4 * 9;

        let mut triplets_jx = TriMat::with_capacity((n_dof, n_dof), nnz_estimate);
        let mut triplets_jv = TriMat::with_capacity((n_dof, n_dof), nnz_estimate);

        // Parallel computation of jacobian contributions
        let contributions: Vec<(Vec<(usize, usize, f64)>, Vec<(usize, usize, f64)>)> =
            self.edge_i.par_iter()
                .enumerate()
                .map(|(edge_idx, &i)| {
                    let j = self.edge_j[edge_idx];
                    let k = self.stiffness[edge_idx];
                    let c = self.damping[edge_idx];
                    let l0 = self.rest_lengths[edge_idx];
                    let link_type = self.link_types[edge_idx];

                    // Compute relative position
                    let dx = self.positions[3*i] - self.positions[3*j];
                    let dy = self.positions[3*i + 1] - self.positions[3*j + 1];
                    let dz = self.positions[3*i + 2] - self.positions[3*j + 2];
                    let l = (dx*dx + dy*dy + dz*dz).sqrt();

                    // Check link type constraints
                    let is_active = match link_type {
                        LinkType::NonCompressive => l >= l0,
                        LinkType::NonTensile => l <= l0,
                        LinkType::Default => true,
                    };

                    let mut jx_triplets = Vec::new();
                    let mut jv_triplets = Vec::new();

                    if !is_active || l < 1e-10 {
                        return (jx_triplets, jv_triplets);
                    }

                    // Unit vector and outer product
                    let ux = dx / l;
                    let uy = dy / l;
                    let uz = dz / l;

                    // Compute stiffness jacobian (3x3 block)
                    // jx = -k * ((l0/l - 1) * (T - I) + T)
                    // where T = u ⊗ u (outer product)

                    // Build 3x3 jacobian block
                    let mut jx_block = [[0.0; 3]; 3];
                    let mut jv_block = [[0.0; 3]; 3];

                    for a in 0..3 {
                        for b in 0..3 {
                            let u = [ux, uy, uz];
                            let outer_prod = u[a] * u[b];
                            let identity = if a == b { 1.0 } else { 0.0 };

                            jx_block[a][b] = -k * ((l0/l - 1.0) * (outer_prod - identity) + outer_prod);
                            jv_block[a][b] = -c * identity;
                        }
                    }

                    // Add contributions to triplet lists
                    // Diagonal blocks (i,i) and (j,j)
                    for a in 0..3 {
                        for b in 0..3 {
                            let val_jx = jx_block[a][b];
                            let val_jv = jv_block[a][b];

                            if val_jx.abs() > 1e-14 {
                                jx_triplets.push((3*i + a, 3*i + b, val_jx));
                                jx_triplets.push((3*j + a, 3*j + b, val_jx));
                            }
                            if val_jv.abs() > 1e-14 {
                                jv_triplets.push((3*i + a, 3*i + b, val_jv));
                                jv_triplets.push((3*j + a, 3*j + b, val_jv));
                            }
                        }
                    }

                    // Off-diagonal blocks (i,j) and (j,i)
                    for a in 0..3 {
                        for b in 0..3 {
                            let val_jx = -jx_block[a][b];
                            let val_jv = -jv_block[a][b];

                            if val_jx.abs() > 1e-14 {
                                jx_triplets.push((3*i + a, 3*j + b, val_jx));
                                jx_triplets.push((3*j + a, 3*i + b, val_jx));
                            }
                            if val_jv.abs() > 1e-14 {
                                jv_triplets.push((3*i + a, 3*j + b, val_jv));
                                jv_triplets.push((3*j + a, 3*i + b, val_jv));
                            }
                        }
                    }

                    (jx_triplets, jv_triplets)
                })
                .collect();

        // Sequential assembly from parallel contributions
        for (jx_trips, jv_trips) in contributions {
            for (row, col, val) in jx_trips {
                triplets_jx.add_triplet(row, col, val);
            }
            for (row, col, val) in jv_trips {
                triplets_jv.add_triplet(row, col, val);
            }
        }

        (triplets_jx.to_csr(), triplets_jv.to_csr())
    }

    /// Get mass matrix
    pub fn mass_matrix(&self) -> &CsMat<f64> {
        self.mass_matrix.as_ref().unwrap()
    }

    /// Update particle positions and velocities
    pub fn update_state(&mut self, positions: Vec<f64>, velocities: Vec<f64>) {
        assert_eq!(positions.len(), self.n_particles * 3);
        assert_eq!(velocities.len(), self.n_particles * 3);

        self.positions = positions;
        self.velocities = velocities;
    }

    /// Get current positions (returns copy)
    pub fn get_positions(&self) -> Vec<f64> {
        self.positions.clone()
    }

    /// Get current velocities (returns copy)
    pub fn get_velocities(&self) -> Vec<f64> {
        self.velocities.clone()
    }

    /// Create constraint mask for filtering constrained DOFs
    pub fn create_constraint_mask(&self) -> Vec<bool> {
        let mut mask = vec![true; self.n_particles * 3];

        for i in 0..self.n_particles {
            if !self.fixed[i] {
                continue;
            }

            match self.constraint_types[i] {
                ConstraintType::Point => {
                    mask[3*i] = false;
                    mask[3*i + 1] = false;
                    mask[3*i + 2] = false;
                }
                ConstraintType::Plane => {
                    let constraint = self.constraints[i];
                    // Fix DOF perpendicular to plane
                    if constraint[0].abs() > 0.999 {
                        mask[3*i] = false;
                    } else if constraint[1].abs() > 0.999 {
                        mask[3*i + 1] = false;
                    } else if constraint[2].abs() > 0.999 {
                        mask[3*i + 2] = false;
                    }
                }
                ConstraintType::Line => {
                    let constraint = self.constraints[i];
                    // Allow movement only along line direction
                    if constraint[0].abs() > 0.999 {
                        mask[3*i + 1] = false;
                        mask[3*i + 2] = false;
                    } else if constraint[1].abs() > 0.999 {
                        mask[3*i] = false;
                        mask[3*i + 2] = false;
                    } else if constraint[2].abs() > 0.999 {
                        mask[3*i] = false;
                        mask[3*i + 1] = false;
                    }
                }
                ConstraintType::Free => {}
            }
        }

        mask
    }
}
