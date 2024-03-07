use ldlt_examples::{
    DenseFaerLdltImpl, LdltSolver, SparseFaerLdltImpl, SprsLdltImpl, SymmetricTripletMatrix,
};

fn main() {
    println!("Hello, world!");

    let b = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let matrix = SymmetricTripletMatrix::example();
    println!(
        "Is semi-positive definite: {}",
        matrix.is_semi_positive_definite()
    );

    let full_matrix = matrix.to_dense();
    println!("Full matrix: {}", full_matrix);

    {
        let sprs_ldlt = SprsLdltImpl::from_triplets(&matrix);
        let l_mat = sprs_ldlt.l_mat();
        let d_mat = sprs_ldlt.d_mat();
        let matrix = sprs_ldlt.to_dense();
        let x = sprs_ldlt.solve(&b);
        println!("L: {}", l_mat);
        println!("D: {}", d_mat);
        println!("Matrix: {}", matrix);
        println!("Matrix: {}", matrix.clone() - full_matrix.clone());
        println!(
            "Matrix: {}",
            l_mat.clone() * nalgebra::DMatrix::from_diagonal(&d_mat) * l_mat.transpose()
        );

        println!("Solved: {}", x);
        let b_from_x_ldlt = matrix * x;
        println!("b from x: {}", b_from_x_ldlt);
    }

    let x_from_qr = full_matrix.clone().qr().solve(&b).unwrap();

    println!("x from qr: {}", x_from_qr);

    println!("b from x: {}", full_matrix.rank(0.000001));

    let b_from_x = full_matrix.clone() * x_from_qr;
    println!("b from x: {}", b_from_x);

    let dense_faer_ldlt = DenseFaerLdltImpl::from_triplets(&matrix);
    let l_mat = dense_faer_ldlt.l_mat();
    let d_mat = dense_faer_ldlt.d_mat();
    let mmatrix = dense_faer_ldlt.to_dense();
    let d_mat = nalgebra::DMatrix::from_diagonal(&d_mat);
    println!("L: {}", l_mat);
    println!("D: {}", d_mat);
    println!("Matrix: {}", mmatrix);
    println!(
        "Matrix: {}",
        l_mat.clone() * d_mat.clone() * l_mat.transpose()
    );
    println!(
        "Matrix: {}",
        l_mat.clone() * d_mat * l_mat.transpose() - full_matrix.clone()
    );

    let x = dense_faer_ldlt.solve(&b);
    println!("Solved: {}", x);
    let sparse_faer_ldlt = SparseFaerLdltImpl::from_triplets(&matrix);
    let x = sparse_faer_ldlt.solve(&b);

    println!("Solved: {}", x);

    println!("matrix: {}", sparse_faer_ldlt.to_dense());
}
