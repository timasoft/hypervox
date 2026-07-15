use criterion::{Criterion, criterion_group, criterion_main};
use hypervox_expr::{VarMap, parse};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct TestVars {
    ndim: usize,
}

impl VarMap for TestVars {
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name {
            "x" => Some(0),
            "y" => Some(if self.ndim > 1 { 1 } else { 0 }),
            "z" => Some(if self.ndim > 2 { 2 } else { 0 }),
            _ => None,
        }
    }
    fn primary_prefix(&self) -> &str {
        "x"
    }
}

const D3: TestVars = TestVars { ndim: 3 };

const FAST_BENCH_SIDE: usize = 128;
const SLOW_BENCH_SIDE: usize = 64;

#[inline(never)]
fn direct_simple(x: f64, y: f64, z: f64) -> f64 {
    x + y * z
}

#[inline(never)]
fn direct_medium(x: f64, y: f64, z: f64) -> f64 {
    x.sin() + y.cos() * (z * z) + (x * x + y * y).sqrt()
}

#[inline(never)]
fn direct_heavy(x: f64, y: f64, z: f64) -> f64 {
    (x.sin() * y.cos()).exp()
        + (z * z + 1.0).ln()
        + ((x * x + y * y + z * z).sqrt()).atan2(black_box(1.0))
        + (x + y).abs().sqrt()
}

#[inline(never)]
fn direct_repeated(x: f64, y: f64, _z: f64) -> f64 {
    (x * x + y * y) * (x * x + y * y) + (x * x + y * y).sin()
}

struct Case {
    name: &'static str,
    hx: &'static str,
    ee: &'static str,
    direct: fn(f64, f64, f64) -> f64,
}

const CASES: &[Case] = &[
    Case {
        name: "simple",
        hx: "x + y * z",
        ee: "x + y * z",
        direct: direct_simple,
    },
    Case {
        name: "medium",
        hx: "sin(x) + cos(y) * z^2 + sqrt(x*x + y*y)",
        ee: "math::sin(x) + math::cos(y) * z^2 + math::sqrt(x*x + y*y)",
        direct: direct_medium,
    },
    Case {
        name: "heavy",
        hx: "exp(sin(x) * cos(y)) + ln(z*z + 1) + atan2(sqrt(x*x + y*y + z*z), 1) + sqrt(abs(x + y))",
        ee: "math::exp(math::sin(x) * math::cos(y)) + math::ln(z*z + 1) + math::atan2(math::sqrt(x*x + y*y + z*z), 1) + math::sqrt(math::abs(x + y))",
        direct: direct_heavy,
    },
    Case {
        name: "repeated",
        hx: "(x*x + y*y)*(x*x + y*y) + sin(x*x + y*y)",
        ee: "(x*x + y*y)*(x*x + y*y) + math::sin(x*x + y*y)",
        direct: direct_repeated,
    },
];

/// A simple triple-nested loop over a `side³` grid, pushing `x, y, z` through
/// `f` and accumulating the results.  Both inputs and output are `black_box`ed
/// to prevent the compiler from constant-folding the entire computation.
fn run_triple_loop<F>(side: usize, scale: f64, mut f: F) -> f64
where
    F: FnMut(f64, f64, f64) -> f64,
{
    let mut sum = 0.0_f64;
    for nz in 0..side {
        let zv = nz as f64 * scale - 5.0;
        for ny in 0..side {
            let yv = ny as f64 * scale - 5.0;
            for nx in 0..side {
                let xv = nx as f64 * scale - 5.0;
                sum += black_box(f(black_box(xv), black_box(yv), black_box(zv)));
            }
        }
    }
    black_box(sum)
}

fn bench_compile_time(c: &mut Criterion, name: &str, expr_src: &str) {
    let mut group = c.benchmark_group(format!("compile/{name}"));

    group.bench_function("parse", |b| {
        b.iter_batched(
            || (),
            |_| black_box(parse(expr_src, &D3).unwrap()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("pre_eval", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).unwrap(),
            |mut node| black_box(node.pre_eval(&[])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("cse", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).unwrap();
                node.pre_eval(&[]);
                node
            },
            |mut node| black_box(node.cse()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("compile", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).unwrap();
                node.pre_eval(&[]);
                node
            },
            |node| black_box(node.compile()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("compile_multi", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).unwrap();
                node.pre_eval(&[]);
                node
            },
            |mut node| black_box(node.compile_multi(&[0, 1, 2])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("prepare", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).unwrap(),
            |mut node| black_box(node.prepare(&[])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("prepare_multi", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).unwrap(),
            |mut node| black_box(node.prepare_multi(&[], &[0, 1, 2])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_expression(
    c: &mut Criterion,
    name: &str,
    expr_src: &str,
    direct_fn: fn(f64, f64, f64) -> f64,
) {
    let side = FAST_BENCH_SIDE;
    let scale = 10.0 / side as f64;

    let mut node_flat = parse(expr_src, &D3).expect("parse failed");
    node_flat.pre_eval(&[]);
    let flat_expr = node_flat.compile();

    let mut node_cse = parse(expr_src, &D3).expect("parse failed");
    let (cse_expr, cse_slots) = node_cse.prepare(&[]);

    let mut node_multi = parse(expr_src, &D3).expect("parse failed");
    let multi = node_multi.prepare_multi(&[], &[0, 1, 2]);

    // Sanity check
    {
        let vars = [1.5, 2.5, -3.0];
        let mut cache = vec![0.0; multi.cse_slots];
        for g in &multi.groups {
            (g.combined)(&vars, &mut cache);
        }
        let compiled_val = (multi.main)(&vars, &mut cache);
        let direct_val = direct_fn(vars[0], vars[1], vars[2]);
        assert!(
            (compiled_val - direct_val).abs() < 1e-12,
            "sanity fail for {name}: compiled={compiled_val}, direct={direct_val}"
        );
    }

    let mut group = c.benchmark_group(name);
    group.sample_size(25);

    group.bench_function("direct", |b| {
        b.iter(|| run_triple_loop(side, scale, direct_fn));
    });

    let mut cache_flat = vec![0.0; 0];
    group.bench_function("flat", |b| {
        let cache = &mut cache_flat;
        b.iter(|| run_triple_loop(side, scale, |x, y, z| flat_expr(&[x, y, z], cache)));
    });

    let mut cache_cse = vec![0.0; cse_slots];
    group.bench_function("cse", |b| {
        let cache = &mut cache_cse;
        b.iter(|| run_triple_loop(side, scale, |x, y, z| cse_expr(&[x, y, z], cache)));
    });

    let mut cache_multi = vec![0.0; multi.cse_slots];
    group.bench_function("multi", |b| {
        let cache = &mut cache_multi;
        b.iter(|| {
            let mut sum = 0.0_f64;
            let mut vars = [0.0; 3];

            for nz in 0..side {
                vars[2] = black_box(nz as f64 * scale - 5.0);
                for g in &multi.groups {
                    if g.level == 2 {
                        (g.combined)(&vars, cache);
                    }
                }

                for ny in 0..side {
                    vars[1] = black_box(ny as f64 * scale - 5.0);
                    for g in &multi.groups {
                        if g.level == 1 {
                            (g.combined)(&vars, cache);
                        }
                    }

                    for nx in 0..side {
                        vars[0] = black_box(nx as f64 * scale - 5.0);
                        sum += black_box((multi.main)(&vars, cache));
                    }
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn register_fast(c: &mut Criterion) {
    for case in CASES {
        bench_expression(c, case.name, case.hx, case.direct);
    }
}

fn register_compile(c: &mut Criterion) {
    for case in CASES {
        bench_compile_time(c, case.name, case.hx);
    }
}

#[cfg_attr(not(feature = "slow-benches"), allow(dead_code))]
mod evalexpr_bench {
    use super::*;
    use evalexpr::{
        ContextWithMutableVariables, DefaultNumericTypes, HashMapContext, Value,
        build_operator_tree,
    };

    fn bench_vs_evalexpr(
        c: &mut Criterion,
        name: &str,
        hx_expr: &str,
        ee_expr: &str,
        direct_fn: fn(f64, f64, f64) -> f64,
    ) {
        let side = SLOW_BENCH_SIDE;
        let scale = 10.0 / side as f64;

        let mut node = parse(hx_expr, &D3).expect("parse failed");
        let (compiled, slots) = node.prepare(&[]);
        let mut cache = vec![0.0; slots];

        let ee_node = build_operator_tree::<DefaultNumericTypes>(ee_expr).expect("evalexpr parse");

        {
            let v = [1.5, 2.5, -3.0];
            let hx = compiled(&v, &mut cache);

            let mut ctx = HashMapContext::new();
            ctx.set_value("x".into(), Value::from_float(v[0])).ok();
            ctx.set_value("y".into(), Value::from_float(v[1])).ok();
            ctx.set_value("z".into(), Value::from_float(v[2])).ok();
            let ee = ee_node
                .eval_with_context(&ctx)
                .unwrap()
                .as_number()
                .unwrap();

            let expected = direct_fn(v[0], v[1], v[2]);
            assert!((hx - expected).abs() < 1e-12, "HX sanity fail for {name}");
            assert!((ee - expected).abs() < 1e-6, "EE sanity fail for {name}");
        }

        let mut group = c.benchmark_group(format!("vs_evalexpr/{name}"));
        group.sample_size(25);

        group.bench_function("hypervox", |b| {
            b.iter(|| run_triple_loop(side, scale, |x, y, z| compiled(&[x, y, z], &mut cache)));
        });

        group.bench_function("evalexpr", |b| {
            let mut ctx = HashMapContext::new();
            b.iter(|| {
                let mut sum = 0.0_f64;
                for nz in 0..side {
                    for ny in 0..side {
                        for nx in 0..side {
                            let xv = black_box(nx as f64 * scale - 5.0);
                            let yv = black_box(ny as f64 * scale - 5.0);
                            let zv = black_box(nz as f64 * scale - 5.0);
                            ctx.set_value("x".into(), Value::from_float(xv)).ok();
                            ctx.set_value("y".into(), Value::from_float(yv)).ok();
                            ctx.set_value("z".into(), Value::from_float(zv)).ok();
                            sum += black_box(
                                ee_node
                                    .eval_with_context_mut(&mut ctx)
                                    .unwrap()
                                    .as_number()
                                    .unwrap(),
                            );
                        }
                    }
                }
                black_box(sum)
            });
        });

        group.finish();
    }

    pub fn register(c: &mut Criterion) {
        for case in CASES {
            bench_vs_evalexpr(c, case.name, case.hx, case.ee, case.direct);
        }
    }
}

criterion_group!(fast_benches, register_fast, register_compile);

#[cfg(feature = "slow-benches")]
criterion_group!(slow_benches, evalexpr_bench::register);

#[cfg(feature = "slow-benches")]
criterion_main!(fast_benches, slow_benches);

#[cfg(not(feature = "slow-benches"))]
criterion_main!(fast_benches);
