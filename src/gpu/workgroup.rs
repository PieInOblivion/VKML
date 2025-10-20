#[derive(Clone, Copy, Debug)]
struct AxisState {
    axis: usize,
    dim: u64,
    limit: u32,
    dispatches: u64,
    local_size: u32,
}

pub fn optimal_workgroup_size(
    max_workgroup_size: [u32; 3],
    max_workgroup_invocations: u32,
    dims: [Option<u64>; 3],
) -> [u32; 3] {
    let mut axes: Vec<AxisState> = dims
        .iter()
        .enumerate()
        .filter_map(|(axis, &maybe_dim)| {
            maybe_dim.filter(|&dim| dim > 0).map(|dim| AxisState {
                axis,
                dim,
                limit: max_workgroup_size[axis],
                dispatches: 1,
                local_size: 1,
            })
        })
        .collect();

    if axes.is_empty() {
        return [1, 1, 1];
    }

    let max_inv = max_workgroup_invocations.max(1);

    for axis in &mut axes {
        let limit = axis.limit.min(max_inv).max(1);
        axis.limit = limit;

        if axis.dim <= limit as u64 {
            axis.local_size = axis.dim as u32;
            axis.dispatches = 1;
        } else {
            axis.dispatches = div_ceil_u64(axis.dim, limit as u64);
            axis.local_size = div_ceil_u32(axis.dim, axis.dispatches).min(limit).max(1);
        }
    }

    let mut product = total_product(&axes);

    while product > max_inv as u64 {
        let mut best: Option<(usize, u64, u64, u32, u64)> = None;

        for (idx, axis) in axes.iter().enumerate() {
            if axis.local_size <= 1 {
                continue;
            }

            let mut next_dispatches = axis.dispatches + 1;
            let mut next_size = div_ceil_u32(axis.dim, next_dispatches)
                .min(axis.limit)
                .max(1);

            while next_size == axis.local_size && next_dispatches < axis.dim {
                next_dispatches += 1;
                next_size = div_ceil_u32(axis.dim, next_dispatches)
                    .min(axis.limit)
                    .max(1);
            }

            if next_size >= axis.local_size {
                continue;
            }

            let new_product = product / axis.local_size.max(1) as u64 * next_size.max(1) as u64;

            let new_waste = projected_waste(&axes, idx, next_size, next_dispatches);

            let candidate = (idx, new_product, next_dispatches, next_size, new_waste);
            match &best {
                None => best = Some(candidate),
                Some((_, best_product, _, best_size, best_waste)) => {
                    if new_product < *best_product
                        || (new_product == *best_product && new_waste < *best_waste)
                        || (new_product == *best_product
                            && new_waste == *best_waste
                            && next_size < *best_size)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }

        let Some((idx, _, next_dispatches, next_size, _)) = best else {
            break;
        };

        axes[idx].dispatches = next_dispatches;
        axes[idx].local_size = next_size.min(axes[idx].limit).max(1);
        product = total_product(&axes);
    }

    let mut result = [1u32; 3];
    for axis in axes {
        result[axis.axis] = axis.local_size;
    }

    result
}

fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return 1;
    }
    value.div_ceil(divisor)
}

fn div_ceil_u32(value: u64, divisor: u64) -> u32 {
    div_ceil_u64(value, divisor).min(u32::MAX as u64) as u32
}

fn axis_waste(dim: u64, local_size: u64, dispatches: u64) -> u64 {
    dispatches.saturating_mul(local_size).saturating_sub(dim)
}

fn projected_waste(axes: &[AxisState], idx: usize, next_size: u32, next_dispatches: u64) -> u64 {
    axes.iter()
        .enumerate()
        .map(|(i, axis)| {
            if i == idx {
                axis_waste(axis.dim, next_size as u64, next_dispatches)
            } else {
                axis_waste(axis.dim, axis.local_size as u64, axis.dispatches)
            }
        })
        .sum()
}

fn total_product(axes: &[AxisState]) -> u64 {
    axes.iter().fold(1u64, |acc, axis| {
        acc.saturating_mul(axis.local_size.max(1) as u64)
    })
}
