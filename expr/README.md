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

## Cargo features

- `slow-benches` -- enables comparison with `evalexpr` in benchmarks (disabled by default)

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/license/mit)

at your option.
