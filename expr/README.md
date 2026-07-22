# hypervox_expr

[![crates.io](https://img.shields.io/crates/v/hypervox_expr.svg)](https://crates.io/crates/hypervox_expr)
[![Documentation](https://docs.rs/hypervox_expr/badge.svg)](https://docs.rs/hypervox_expr)

High-performance mathematical expression parser, AST optimizer, and closure compiler for N-dimensional evaluation.

## Features

- **Pratt parser** -- parses string expressions into an AST with standard operator precedence and right-associative `^`/`**`
- **Constant folding and algebraic simplification** (`pre_eval`) -- evaluates constant sub-expressions, identity elimination, zero propagation, negation rewrites, inverse composition, constant reassociation
- **Common subexpression elimination** (`cse`) -- extracts repeated subtrees into shared slots evaluated once
- **Closure compilation** (`compile`) -- emits fused MulAdd/MulSub/NegMulAdd/NegMul patterns when detected
- **Multi-level invariant hoisting** (`compile_multi`) -- extracts dimension-invariant sub-expressions into hierarchical pre-computed closures for efficient N-dimensional grid evaluation
- **Extensible** -- add custom constants and functions via `ExtF0`/`ExtF1`/`ExtF2` traits and `define_ext_f{0,1,2}!` macros
- **Variable resolution** -- implement `VarMap` to define spatial aliases (`x`, `y`, `z`) and indexed variables (`x0`, `x1`, ...)

## Built-in functions

- **Constants**: `PI`, `E`
- **Unary**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sqrt`, `cbrt`, `exp`, `ln`, `log10`, `log2`, `floor`, `ceil`, `round`, `trunc`, `abs`
- **Binary**: `atan2`, `pow`
- **Operators**: `+`, `-`, `*`, `/`, `^`/`**` (power, right-associative), `%` (modulo), `|x|` (pipe-abs)

`0^0` evaluates to `1`.

## Quick start

```rust
use hypervox_expr::{parse, VarMap};

struct MyVars;
impl VarMap for MyVars {
    fn ndim(&self) -> usize { 3 }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None }
    }
    fn primary_prefix(&self) -> &str { "x" }
}

let mut node = parse("sin(x) * cos(y) + z", &MyVars).unwrap();
let (f, slots) = node.prepare(&[]);
let mut cache = vec![0.0; slots];
let result = f(&[1.0, 2.0, 3.0], &mut cache);
```

## Multi-level evaluation

```rust
let mut node = parse("cos(z)*tan(y) + x", &MyVars).unwrap();
let multi = node.prepare_multi(&[], &[0, 1, 2]);
let mut cache = vec![0.0; multi.cse_slots];
for g in &multi.groups {
    (g.combined)(&[1.0, 2.0, 3.0], &mut cache);
}
let result = (multi.main)(&[1.0, 2.0, 3.0], &mut cache);
```

## Extending with custom functions

```rust
use hypervox_expr::{parse_with_ext, NoExtF};
hypervox_expr::define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);

let mut node = parse_with_ext::<_, NoExtF, MyF1, NoExtF>("cube(x)", &MyVars).unwrap();
let (f, slots) = node.prepare(&[]);
assert_eq!(f(&[3.0], &mut vec![0.0; slots]), 27.0);
```

## Performance

The `compile` strategy emits a single flat closure; `compile_multi` hoists
dimension-invariant sub-expressions into pre-computed groups, trading longer
compile time for faster evaluation on large grids:

| Benchmark  | Parse   | Pre-eval | Compile | Compile multi | CSE     | Prepare | Prepare multi |
|------------|---------|----------|---------|---------------|---------|---------|---------------|
| simple     |  0.53µs | 0.03µs   |  0.13µs |   0.80µs      |  0.05µs |  0.24µs |   0.90µs      |
| medium     |  1.63µs | 0.15µs   |  0.60µs |   2.30µs      |  0.40µs |  1.21µs |   2.97µs      |
| heavy      |  3.14µs | 0.24µs   |  1.21µs |   4.86µs      |  1.57µs |  3.07µs |   7.00µs      |
| repeated   |  2.16µs | 0.14µs   |  0.72µs |   2.99µs      |  1.14µs |  2.09µs |   4.06µs      |
| very_heavy | 34.96µs | 2.76µs   | 15.51µs | 132.13µs      | 55.62µs | 75.54µs | 244.33µs      |

Runtime evaluation on a 128^3 grid comparing compilation strategies:

| Benchmark  | direct  | flat      | cse       | multi     |
|------------|---------|-----------|-----------|-----------|
| simple     |   9.2ms |    19.3ms |    19.3ms |    13.4ms |
| medium     |  54.2ms |   103.1ms |   102.6ms |    66.1ms |
| heavy      | 120.1ms |   239.1ms |   241.6ms |   170.1ms |
| repeated   |  31.6ms |   111.7ms |   100.5ms |    86.0ms |
| very_heavy | 753.5ms | 3,242.6ms | 2,274.7ms | 1,885.3ms |

vs [`evalexpr`](https://crates.io/crates/evalexpr) on a 64^3 grid:

| Benchmark  | hypervox_expr | evalexpr  | speedup  |
|------------|---------------|-----------|----------|
| simple     |   2.4ms       |   104.7ms | **~44×** |
| medium     |  12.9ms       |   295.4ms | **~23×** |
| heavy      |  30.4ms       |   598.5ms | **~20×** |
| repeated   |  12.2ms       |   385.5ms | **~32×** |
| very_heavy | 293.2ms       | 6,446.4ms | **~22×** |

Measurements from [`criterion`] benchmarks on GitHub Actions
(ubuntu-latest) at commit `e854400d`. [View live dashboard][bencher]

[`criterion`]: https://github.com/criterion-rs/criterion.rs
[bencher]: https://bencher.dev/perf/hypervox-expr

## Cargo features

- `slow-benches` -- enables comparison with `evalexpr` in benchmarks (disabled by default)

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/license/mit)

at your option.
