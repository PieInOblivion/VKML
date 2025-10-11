/// CPU implementation of GEMM for f32
/// Computes Y = alpha * op(A) * op(B) + beta * C
/// where op(X) is either X or X^T depending on transpose flags
pub fn f32_cpu(
    a_dims: Vec<usize>,
    b_dims: Vec<usize>,
    y_dims: Vec<usize>,
    a_bytes: &[u8],
    b_bytes: &[u8],
    c_bytes: Option<&[u8]>,
    y_bytes: &mut [u8],
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) {
    assert_eq!(a_dims.len(), 2, "GEMM: A must be 2D");
    assert_eq!(b_dims.len(), 2, "GEMM: B must be 2D");
    assert_eq!(y_dims.len(), 2, "GEMM: Y must be 2D");

    // Cast byte slices to f32 slices
    let a_data =
        unsafe { std::slice::from_raw_parts(a_bytes.as_ptr() as *const f32, a_bytes.len() / 4) };
    let b_data =
        unsafe { std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, b_bytes.len() / 4) };
    let y_data = unsafe {
        std::slice::from_raw_parts_mut(y_bytes.as_mut_ptr() as *mut f32, y_bytes.len() / 4)
    };

    // Determine dimensions
    // A is (M, K) or (K, M) if trans_a
    // B is (K, N) or (N, K) if trans_b
    // Y is (M, N)
    let (m, k_a) = if trans_a {
        (a_dims[1], a_dims[0])
    } else {
        (a_dims[0], a_dims[1])
    };

    let (k_b, n) = if trans_b {
        (b_dims[1], b_dims[0])
    } else {
        (b_dims[0], b_dims[1])
    };

    assert_eq!(k_a, k_b, "GEMM: K dimensions must match");
    let k = k_a;

    assert_eq!(y_dims[0], m, "GEMM: Y row dimension must equal M");
    assert_eq!(y_dims[1], n, "GEMM: Y col dimension must equal N");

    // Helper to get element from A (considering transpose)
    let get_a = |row: usize, col: usize| -> f32 {
        let idx = if trans_a {
            // A is stored as (K, M), we want A[row][col] = A_stored[col][row]
            col * a_dims[1] + row
        } else {
            // A is stored as (M, K), we want A[row][col]
            row * a_dims[1] + col
        };
        a_data[idx]
    };

    // Helper to get element from B (considering transpose)
    let get_b = |row: usize, col: usize| -> f32 {
        let idx = if trans_b {
            // B is stored as (N, K), we want B[row][col] = B_stored[col][row]
            col * b_dims[1] + row
        } else {
            // B is stored as (K, N), we want B[row][col]
            row * b_dims[1] + col
        };
        b_data[idx]
    };

    // Initialize Y with beta * C if C is provided, otherwise zero
    if let Some(c_bytes_slice) = c_bytes {
        let c_data = unsafe {
            std::slice::from_raw_parts(
                c_bytes_slice.as_ptr() as *const f32,
                c_bytes_slice.len() / 4,
            )
        };

        // C should be broadcastable to (M, N)
        // For simplicity, we'll assume C is either:
        // - scalar (1 element)
        // - 1D with N elements (broadcast across rows)
        // - 2D with shape (M, N)
        let c_len = c_data.len();

        for i in 0..m {
            for j in 0..n {
                let c_val = if c_len == 1 {
                    // Scalar
                    c_data[0]
                } else if c_len == n {
                    // 1D array broadcast across rows
                    c_data[j]
                } else if c_len == m * n {
                    // Full 2D array
                    c_data[i * n + j]
                } else {
                    panic!("GEMM: C tensor shape not broadcastable to ({}, {})", m, n);
                };

                y_data[i * n + j] = beta * c_val;
            }
        }
    } else {
        // No C tensor, initialize Y to zero
        for val in y_data.iter_mut() {
            *val = 0.0;
        }
    }

    // Compute Y += alpha * A * B
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += get_a(i, p) * get_b(p, j);
            }
            y_data[i * n + j] += alpha * sum;
        }
    }
}
