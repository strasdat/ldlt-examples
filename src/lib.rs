#![deny(missing_docs)]

//! Ldlt solver implementations

use nalgebra::{DMatrix, DVector};
use nshare::ToNalgebra;
use sprs_ldl::LdlNumeric;

/// A matrix in triplet format
pub struct SymmetricTripletMatrix {
    /// upper diagonal triplets
    pub upper_triplets: Vec<(usize, usize, f64)>,
    /// row count (== column count)
    pub size: usize,
}

impl SymmetricTripletMatrix {
    /// Create an example matrix
    pub fn example() -> Self {
        Self {
            upper_triplets: vec![
                (0, 0, 3.05631771),
                (1, 1, 60.05631771),
                (2, 2, 6.05631771),
                (3, 3, 5.05631771),
                (4, 4, 8.05631771),
                (5, 5, 5.05631771),
                (6, 6, 0.05631771),
                (7, 7, 10.005631771),
                (0, 1, 2.41883573),
                (0, 3, 2.41883573),
                (0, 5, 1.88585946),
                (1, 3, 1.73897015),
                (1, 5, 2.12387697),
                (1, 7, 1.47609157),
                (2, 7, 1.4541327),
                (3, 4, 2.35666066),
                (3, 7, 0.94642903),
            ],
            size: 8,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut full_matrix = nalgebra::DMatrix::from_element(self.size, self.size, 0.0);
        for &(row, col, value) in self.upper_triplets.iter() {
            full_matrix[(row, col)] = value;
            if row != col {
                full_matrix[(col, row)] = value;
            }
        }
        full_matrix
    }

    /// Returns true if the matrix is semi-positive definite
    pub fn is_semi_positive_definite(&self) -> bool {
        let full_matrix = self.to_dense();
        let eigen_comp = full_matrix.symmetric_eigen();
        eigen_comp.eigenvalues.iter().all(|&x| x >= 0.0)
    }
}

/// Solves a linear system of equations using the LDLT decomposition
pub trait LdltSolver {
    /// Create a new solver from triplets
    fn from_triplets(sym_tri_mat: &SymmetricTripletMatrix) -> Self;

    /// Convert to dense matrix
    fn to_dense(&self) -> nalgebra::DMatrix<f64>;

    /// Solve the linear system: Ax = b
    /// Returns the solution x
    fn solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64>;
}

/// A matrix in triplet format for sparse LDLT factorization / using sprs crate
pub struct SprsLdltImpl {
    tri_mat: sprs::TriMat<f64>,
}

impl SprsLdltImpl {
    fn to_ldlt(&self) -> LdlNumeric<f64, usize> {
        let csr = self.tri_mat.to_csr();
        let ldl = sprs_ldl::Ldl::new().check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry);
        ldl.numeric(csr.view()).unwrap()
    }

    /// Returns the dense L matrix - for testing only
    pub fn l_mat(&self) -> nalgebra::DMatrix<f64> {
        self.to_ldlt().l().to_csc().to_dense().into_nalgebra()
            + nalgebra::DMatrix::identity(self.tri_mat.cols(), self.tri_mat.cols())
    }

    /// Returns the dense D vector - for testing only
    pub fn d_mat(&self) -> nalgebra::DVector<f64> {
        DVector::from_iterator(self.tri_mat.cols(), self.to_ldlt().d().iter().copied())
    }
}

impl LdltSolver for SprsLdltImpl {
    /// Create a new solver from triplets - using sprs crate
    fn from_triplets(sym_tri_mat: &SymmetricTripletMatrix) -> Self {
        let mut tri_mat = sprs::TriMat::new((sym_tri_mat.size, sym_tri_mat.size));
        for &(row, col, value) in sym_tri_mat.upper_triplets.iter() {
            tri_mat.add_triplet(row, col, value);
            if row != col {
                tri_mat.add_triplet(col, row, value);
            }
        }
        Self { tri_mat }
    }

    /// Convert to dense matrix
    fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        self.tri_mat.to_csr::<usize>().to_dense().into_nalgebra()
    }

    /// Solve the linear system: Ax = b
    /// Returns the solution x
    fn solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(
            self.to_ldlt()
                .solve(b.iter().copied().collect::<Vec<f64>>()),
        )
    }
}

/// A symmetric matrix for LDLT factorization / using faer crate
pub struct DenseFaerLdltImpl {
    upper_dense: faer::Mat<f64>,
}

impl DenseFaerLdltImpl {
    fn to_ldlt(&self) -> faer::Mat<f64> {
        let mut ldlt_mat = self.upper_dense.clone();
        faer::linalg::cholesky::ldlt_diagonal::compute::raw_cholesky_in_place(
            ldlt_mat.as_mut(),
            Default::default(),
            faer::Parallelism::Rayon(8),
            faer::dyn_stack::PodStack::new(&mut faer::dyn_stack::GlobalPodBuffer::new(
                faer::linalg::cholesky::ldlt_diagonal::compute::raw_cholesky_in_place_req::<f64>(
                    self.upper_dense.ncols(),
                    faer::Parallelism::Rayon(8),
                    Default::default(),
                )
                .unwrap(),
            )),
            Default::default(),
        );
        ldlt_mat
    }

    fn to_dense(mat: faer::Mat<f64>) -> nalgebra::DMatrix<f64> {
        let mut nalgebra_mat = DMatrix::<f64>::zeros(mat.nrows(), mat.ncols());
        for row in 0..mat.nrows() {
            for col in 0..mat.ncols() {
                *nalgebra_mat.get_mut((row, col)).unwrap() = *mat.get(row, col);
            }
        }
        nalgebra_mat
    }

    /// Returns the dense L matrix - for testing only
    pub fn l_mat(&self) -> nalgebra::DMatrix<f64> {
        let raw_ldlt = self.to_ldlt();

        let mut l_mat =
            nalgebra::DMatrix::zeros(self.upper_dense.nrows(), self.upper_dense.ncols());

        for row in 0..l_mat.nrows() {
            for col in 0..l_mat.ncols() {
                if row > col {
                    *l_mat.get_mut((row, col)).unwrap() = *raw_ldlt.get(row, col);
                } else if row == col {
                    *l_mat.get_mut((row, col)).unwrap() = 1.0;
                }
            }
        }
        l_mat
    }

    /// Returns the dense D vector - for testing only
    pub fn d_mat(&self) -> nalgebra::DVector<f64> {
        let raw_ldlt = self.to_ldlt();
        let mut d_vec = DVector::zeros(self.upper_dense.ncols());
        for i in 0..self.upper_dense.ncols() {
            *d_vec.get_mut(i).unwrap() = 1.0 / *raw_ldlt.get(i, i);
        }
        d_vec
    }
}

impl LdltSolver for DenseFaerLdltImpl {
    fn from_triplets(sym_tri_mat: &SymmetricTripletMatrix) -> Self {
        let mut upper_dense = faer::Mat::<f64>::zeros(sym_tri_mat.size, sym_tri_mat.size);
        for &(row, col, value) in sym_tri_mat.upper_triplets.iter() {
            *upper_dense.get_mut(row, col) = value;
            if row != col {
                *upper_dense.get_mut(col, row) = value;
            }
        }
        Self { upper_dense }
    }

    fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        Self::to_dense(self.upper_dense.clone())
    }

    fn solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        let mut x = faer::Mat::<f64>::zeros(b.len(), 1);
        for i in 0..b.len() {
            *x.get_mut(i, 0) = *b.get(i).unwrap();
        }
        let ldlt = self.to_ldlt();
        faer::linalg::cholesky::ldlt_diagonal::solve::solve_in_place_with_conj(
            ldlt.as_ref(),
            faer::Conj::No,
            x.as_mut(),
            faer::Parallelism::Rayon(8),
            faer::dyn_stack::PodStack::new(&mut []),
        );

        let mut x_nalgebra = DVector::zeros(b.len());
        for i in 0..b.len() {
            *x_nalgebra.get_mut(i).unwrap() = *x.get(i, 0);
        }
        x_nalgebra
    }
}

/// A matrix in triplet format for sparse LDLT factorization / using faer crate
pub struct SparseFaerLdltImpl {
    upper_ccs: faer::sparse::SparseColMat<usize, f64>,
}

struct SparseFaerLdltPerm {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl SparseFaerLdltPerm {
    fn perm_upper_ccs(
        &self,
        upper_ccs: &faer::sparse::SparseColMat<usize, f64>,
    ) -> faer::sparse::SparseColMat<usize, f64> {
        let dim = upper_ccs.ncols();
        let nnz = upper_ccs.compute_nnz();
        let perm_ref = unsafe { faer::perm::PermRef::new_unchecked(&self.perm, &self.perm_inv) };

        let mut ccs_perm_col_ptrs = Vec::new();
        let mut ccs_perm_row_indices = Vec::new();
        let mut ccs_perm_values = Vec::new();

        ccs_perm_col_ptrs.try_reserve_exact(dim + 1).unwrap();
        ccs_perm_col_ptrs.resize(dim + 1, 0usize);
        ccs_perm_row_indices.try_reserve_exact(nnz).unwrap();
        ccs_perm_row_indices.resize(nnz, 0usize);
        ccs_perm_values.try_reserve_exact(nnz).unwrap();
        ccs_perm_values.resize(nnz, 0.0f64);

        let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
            faer::sparse::utils::permute_hermitian_req::<usize>(dim).unwrap(),
        )
        .unwrap();
        faer::sparse::utils::permute_hermitian::<usize, f64>(
            &mut ccs_perm_values,
            &mut ccs_perm_col_ptrs,
            &mut ccs_perm_row_indices,
            upper_ccs.as_ref(),
            perm_ref,
            faer::Side::Upper,
            faer::Side::Upper,
            faer::dyn_stack::PodStack::new(&mut mem),
        );

        faer::sparse::SparseColMat::<usize, f64>::new(
            unsafe {
                faer::sparse::SymbolicSparseColMat::new_unchecked(
                    dim,
                    dim,
                    ccs_perm_col_ptrs,
                    None,
                    ccs_perm_row_indices,
                )
            },
            ccs_perm_values,
        )
    }
}

/// Symbolic LDLT factorization and permutation
struct SparseFaerLdltSymbolicPerm {
    perm: SparseFaerLdltPerm,
    symbolic: faer::sparse::linalg::cholesky::simplicial::SymbolicSimplicialCholesky<usize>,
}

impl SparseFaerLdltImpl {
    fn symbolic_and_perm(&self) -> SparseFaerLdltSymbolicPerm {
        let dim = self.upper_ccs.ncols();

        let nnz = self.upper_ccs.compute_nnz();

        let (perm, perm_inv) = {
            let mut perm = Vec::new();
            let mut perm_inv = Vec::new();
            perm.try_reserve_exact(dim).unwrap();
            perm_inv.try_reserve_exact(dim).unwrap();
            perm.resize(dim, 0usize);
            perm_inv.resize(dim, 0usize);

            let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
                faer::sparse::linalg::amd::order_req::<usize>(dim, nnz).unwrap(),
            )
            .unwrap();
            faer::sparse::linalg::amd::order(
                &mut perm,
                &mut perm_inv,
                self.upper_ccs.symbolic(),
                faer::sparse::linalg::amd::Control::default(),
                faer::dyn_stack::PodStack::new(&mut mem),
            )
            .unwrap();

            (perm, perm_inv)
        };
        let perm_struct = SparseFaerLdltPerm { perm, perm_inv };

        let ccs_perm_upper = perm_struct.perm_upper_ccs(&self.upper_ccs);

        let symbolic = {
            let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(
                faer::dyn_stack::StackReq::try_any_of([
                    faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky_req::<
                        usize,
                    >(dim, self.upper_ccs.compute_nnz()).unwrap(),
                    faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_symbolic_req::<
                        usize,
                    >(dim).unwrap(),
                ]).unwrap(),
            ).unwrap();
            let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

            let mut etree = Vec::new();
            let mut col_counts = Vec::new();
            etree.try_reserve_exact(dim).unwrap();
            etree.resize(dim, 0isize);
            col_counts.try_reserve_exact(dim).unwrap();
            col_counts.resize(dim, 0usize);

            faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                ccs_perm_upper.symbolic(),
                faer::reborrow::ReborrowMut::rb_mut(&mut stack),
            );
            faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_symbolic(
                ccs_perm_upper.symbolic(),
                // SAFETY: `etree` was filled correctly by
                // `simplicial::prefactorize_symbolic_cholesky`.
                unsafe {
                    faer::sparse::linalg::cholesky::simplicial::EliminationTreeRef::from_inner(
                        &etree,
                    )
                },
                &col_counts,
                faer::reborrow::ReborrowMut::rb_mut(&mut stack),
            )
            .unwrap()
        };
        SparseFaerLdltSymbolicPerm {
            perm: perm_struct,
            symbolic,
        }
    }

    fn solve_from_symbolic(
        &self,
        b: &nalgebra::DVector<f64>,
        symbolic_perm: &SparseFaerLdltSymbolicPerm,
    ) -> nalgebra::DVector<f64> {
        let dim = self.upper_ccs.ncols();
        let perm_ref = unsafe {
            faer::perm::PermRef::new_unchecked(
                &symbolic_perm.perm.perm,
                &symbolic_perm.perm.perm_inv,
            )
        };
        let symbolic = &symbolic_perm.symbolic;

        let mut mem = faer::dyn_stack::GlobalPodBuffer::try_new(faer::dyn_stack::StackReq::try_all_of([
            faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt_req::<usize, f64>(dim).unwrap(),
            faer::perm::permute_rows_in_place_req::<usize, f64>(dim, 1).unwrap(),
            symbolic.solve_in_place_req::<f64>(dim).unwrap(),
        ]).unwrap()).unwrap();
        let mut stack = faer::dyn_stack::PodStack::new(&mut mem);

        let mut l_values = Vec::new();
        l_values.try_reserve_exact(symbolic.len_values()).unwrap();
        l_values.resize(symbolic.len_values(), 0.0f64);
        let ccs_perm_upper = symbolic_perm.perm.perm_upper_ccs(&self.upper_ccs);

        faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
            &mut l_values,
            ccs_perm_upper.as_ref(),
            faer::sparse::linalg::cholesky::LdltRegularization::default(),
            &symbolic,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );

        let ldlt =
            faer::sparse::linalg::cholesky::simplicial::SimplicialLdltRef::<'_, usize, f64>::new(
                &symbolic, &l_values,
            );

        let mut x = b.clone();
        let mut binding = x.as_mut_slice();
        let mut x_ref = faer::mat::from_column_major_slice_mut::<f64>(&mut binding, dim, 1);
        faer::perm::permute_rows_in_place(
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            perm_ref,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            faer::Parallelism::None,
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );
        faer::perm::permute_rows_in_place(
            faer::reborrow::ReborrowMut::rb_mut(&mut x_ref),
            perm_ref.inverse(),
            faer::reborrow::ReborrowMut::rb_mut(&mut stack),
        );

        x
    }
}

impl LdltSolver for SparseFaerLdltImpl {
    fn from_triplets(sym_tri_mat: &SymmetricTripletMatrix) -> Self {
        SparseFaerLdltImpl {
            upper_ccs: faer::sparse::SparseColMat::try_new_from_triplets(
                sym_tri_mat.size,
                sym_tri_mat.size,
                &sym_tri_mat.upper_triplets,
            )
            .unwrap(),
        }
    }

    fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let upper_dense = self.upper_ccs.to_dense();
        let mut nalgebra_mat = DMatrix::<f64>::zeros(upper_dense.nrows(), upper_dense.ncols());
        for row in 0..upper_dense.nrows() {
            for col in row..upper_dense.ncols() {
                let value = *upper_dense.get(row, col);
                *nalgebra_mat.get_mut((row, col)).unwrap() = value;
                *nalgebra_mat.get_mut((col, row)).unwrap() = value;
            }
        }
        nalgebra_mat
    }

    fn solve(&self, b: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        let symbolic_perm = self.symbolic_and_perm();
        self.solve_from_symbolic(b, &symbolic_perm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldlt() {
        let sym_tri_mat = SymmetricTripletMatrix::example();
        assert!(sym_tri_mat.is_semi_positive_definite());
        let dense_mat = sym_tri_mat.to_dense();

        let sprs_mat = SprsLdltImpl::from_triplets(&sym_tri_mat);
        assert_eq!(sprs_mat.to_dense(), dense_mat);

        let faer_dense_mat = DenseFaerLdltImpl::from_triplets(&sym_tri_mat);
        assert_eq!(faer_dense_mat.to_dense(), dense_mat);

        let faer_sparse_mat = SparseFaerLdltImpl::from_triplets(&sym_tri_mat);
        assert_eq!(faer_sparse_mat.to_dense(), dense_mat);

        let b = DVector::from_element(8, 1.0);
        let x_sprs = sprs_mat.solve(&b);
        let x_faer_dense = faer_dense_mat.solve(&b);
        let x_faer_sparse = faer_sparse_mat.solve(&b);

        approx::assert_abs_diff_eq!(dense_mat.clone() * x_sprs.clone(), b);

        approx::assert_abs_diff_eq!(x_faer_dense, x_sprs);
        approx::assert_abs_diff_eq!(x_faer_sparse, x_sprs);
    }
}
