//! Math expression parser, compiler, and CSE engine.
//!
//! Parses string expressions (e.g. `"sin(x)*cos(y)"`) into an AST ([`Node`]),
//! applies constant folding ([`Node::pre_eval`]),
//! common-subexpression elimination ([`Node::cse`]),
//! and compiles to closures ([`Node::compile`] / [`Node::compile_multi`])
//! for fast repeated evaluation over N-dimensional grids.
//! Variable names are resolved through the [`VarMap`] trait.

use std::{
    fmt::{self, Display},
    str::FromStr,
};

pub mod index_set;
pub use index_set::{
    ArithIndexSet, ArithIndexSetTryFromError, ArithRangeFrom, ArithRangeIter, IndexSet,
    IndexSetIter,
};

/// Structured error type for expression parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Lexer { col: usize, kind: LexerErrorKind },
    Parser { col: usize, kind: ParseErrorKind },
}

/// Lexical analysis error details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexerErrorKind {
    UnexpectedChar(char),
    InvalidNumber(String),
}

/// Parsing error details.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    EmptyExpression,
    TrailingToken(Token),
    ExpectedRParen(Token),
    ExpectedPipe(Token),
    UnexpectedToken(Token),
    FunctionArgCount {
        name: String,
        expected: usize,
        found: usize,
    },
    ExpectedRParenOrComma(Token),
    VarOutOfRange {
        name: String,
        max: usize,
    },
    UnknownIdentifier(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Lexer { col, kind } => write!(f, "at column {col}: {kind}"),
            Error::Parser { col, kind } => write!(f, "at column {col}: {kind}"),
        }
    }
}

impl Display for LexerErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerErrorKind::UnexpectedChar(c) => write!(f, "unexpected character '{c}'"),
            LexerErrorKind::InvalidNumber(s) => write!(f, "invalid number '{s}'"),
        }
    }
}

impl Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::EmptyExpression => write!(f, "expression cannot be empty"),
            ParseErrorKind::TrailingToken(tok) => {
                write!(f, "unexpected token {tok} after expression")
            }
            ParseErrorKind::ExpectedRParen(tok) => {
                write!(f, "expected ')' but found {tok}")
            }
            ParseErrorKind::ExpectedPipe(tok) => {
                write!(f, "expected '|' but found {tok}")
            }
            ParseErrorKind::UnexpectedToken(tok) => {
                write!(f, "unexpected token {tok}")
            }
            ParseErrorKind::FunctionArgCount {
                name,
                expected,
                found,
            } => {
                let s = if *expected == 1 {
                    "argument"
                } else {
                    "arguments"
                };
                write!(
                    f,
                    "function '{name}' requires {expected} {s}, but found {found}"
                )
            }
            ParseErrorKind::ExpectedRParenOrComma(tok) => {
                write!(f, "expected ')' or ',' but found {tok}")
            }
            ParseErrorKind::VarOutOfRange { name, max } => {
                write!(f, "variable '{name}' out of range: max index is {max}")
            }
            ParseErrorKind::UnknownIdentifier(name) => {
                write!(f, "unknown identifier '{name}'")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Trait for resolving variable names in expressions.
/// Implementors define variable aliases and the indexed-variable naming scheme.
pub trait VarMap {
    /// Number of dimensions.
    fn ndim(&self) -> usize;

    /// Resolve a variable alias (e.g. "x", "y", "z") to a dimension index.
    fn resolve_alias(&self, name: &str) -> Option<usize>;

    /// Primary variable prefix used for indexed variables (e.g. "x0", "x1", ..).
    fn primary_prefix(&self) -> &str;
}

/// AST node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Num(f64),
    Var(usize),
    Neg(Box<Node>),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
    Pow(Box<Node>, Box<Node>),
    Mod(Box<Node>, Box<Node>),
    F1(F1, Box<Node>),
    F2(F2, Box<Node>, Box<Node>),
    /// let slot_i = expr in body
    Let(usize, Box<Node>, Box<Node>),
    /// reference to cached CSE slot
    CseRef(usize, IndexSet),
}

/// A compiled expression closure: `(vars, cse_cache) -> result`.
pub type CompiledExpr = Box<dyn Fn(&[f64], &mut [f64]) -> f64 + Send + Sync>;

macro_rules! define_f0 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Constants (PI, E).
        #[derive(Debug, Clone, Copy)]
        pub enum F0 {
            $($variant,)*
        }
        impl F0 {
            /// Evaluate the constant as an f64.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F0;
            /// let pi = F0::PI.to_num();
            /// assert!((pi - std::f64::consts::PI).abs() < 1e-15);
            /// ```
            pub fn to_num(self) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All constant names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F0 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown const '{s}'")),
                }
            }
        }
    };
}

macro_rules! define_f1 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Single-argument math functions.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum F1 {
            $($variant,)*
        }
        impl F1 {
            /// Resolve to a function pointer.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F1;
            /// let f = F1::Sin.to_fn();
            /// assert!((f(1.0) - 1.0_f64.sin()).abs() < 1e-15);
            /// ```
            #[inline]
            pub fn to_fn(self) -> fn(f64) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F1 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

macro_rules! define_f2 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Two-argument math functions (atan2, pow).
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum F2 {
            $($variant,)*
        }
        impl F2 {
            /// Resolve to a function pointer.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F2;
            /// let atan2 = F2::Atan2.to_fn();
            /// assert_eq!(atan2(0.0, 1.0), 0.0);
            /// ```
            #[inline]
            pub fn to_fn(self) -> fn(f64, f64) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F2 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

define_f0! {
    PI => "PI" = std::f64::consts::PI,
    E => "E" = std::f64::consts::E,
}

define_f1! {
    Sin => "sin" = f64::sin,
    Cos => "cos" = f64::cos,
    Tan => "tan" = f64::tan,
    Asin => "asin" = f64::asin,
    Acos => "acos" = f64::acos,
    Atan => "atan" = f64::atan,
    Sinh => "sinh" = f64::sinh,
    Cosh => "cosh" = f64::cosh,
    Tanh => "tanh" = f64::tanh,
    Sqrt => "sqrt" = f64::sqrt,
    Cbrt => "cbrt" = f64::cbrt,
    Exp => "exp" = f64::exp,
    Ln => "ln" = f64::ln,
    Log10 => "log10" = f64::log10,
    Log2 => "log2" = f64::log2,
    Floor => "floor" = f64::floor,
    Ceil => "ceil" = f64::ceil,
    Round => "round" = f64::round,
    Trunc => "trunc" = f64::trunc,
    Abs => "abs" = f64::abs,
}

define_f2! {
    Atan2 => "atan2" = f64::atan2,
    Pow => "pow" = |a, b| {
        if a == 0.0 && b == 0.0 {
            1.0
        } else {
            let exp_int = b as i32;
            if (exp_int as f64) == b {
                a.powi(exp_int)
            } else {
                a.powf(b)
            }
        }
    },
}

/// Return a comma-separated list of available constant names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f0_list;
/// let list = f0_list();
/// assert!(list.contains("PI"));
/// ```
pub fn f0_list() -> String {
    F0::NAMES.join(", ")
}

/// Return a comma-separated list of available single-argument function names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f1_list;
/// let list = f1_list();
/// assert!(list.contains("sin"));
/// ```
pub fn f1_list() -> String {
    F1::NAMES.join(", ")
}

/// Return a comma-separated list of available two-argument function names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f2_list;
/// let list = f2_list();
/// assert!(list.contains("atan2"));
/// ```
pub fn f2_list() -> String {
    F2::NAMES.join(", ")
}

/// A group of dimension-invariant sub-expressions at a given nesting level.
pub struct InvariantGroup {
    /// Nesting level: higher = more invariant (evaluated sooner).
    pub level: usize,
    /// Combined closure populating CSE slots for all invariants at this level.
    pub combined: CompiledExpr,
}

/// Multi-level compiled expression with invariant groups and main expression.
pub struct CompiledExprMulti {
    /// Invariant groups.
    pub groups: Vec<InvariantGroup>,
    /// Main expression after extracting all invariants.
    pub main: CompiledExpr,
    /// Total number of CSE slots required.
    pub cse_slots: usize,
}

impl Node {
    /// Evaluate constant sub-expressions and apply algebraic simplifications at compile time.
    ///
    /// When `vars[i]` is `Some(v)`, variable `i` is replaced with `v` before folding.
    ///
    /// Simplifications include:
    /// identity elimination (`x+0`, `x*1`, `x^1`),
    /// zero propagation (`x*0`),
    /// negation rewrites (`-(-x)` -> `x`, `x/-1` -> `-x`),
    /// inverse composition (`ln(exp(x))` -> `x`),
    /// constant reassociation,
    /// and `x-x` -> `0`, `x/x` -> `1`, `x+x` -> `2*x`.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let mut node = Node::Add(Box::new(Node::Num(2.0)), Box::new(Node::Num(3.0)));
    /// node.pre_eval(&[]);
    /// assert_eq!(node, Node::Num(5.0));
    /// ```
    pub fn pre_eval(&mut self, vars: &[Option<f64>]) {
        match self {
            Node::Num(_) => {}
            Node::Var(i) => {
                if let Some(Some(v)) = vars.get(*i) {
                    *self = Node::Num(*v);
                }
            }
            Node::Neg(a) => {
                a.pre_eval(vars);
                if let Node::Num(x) = a.as_ref() {
                    *self = Node::Num(-*x);
                } else if let Node::Neg(inner) = a.as_ref() {
                    // -(-x) => x
                    *self = *inner.clone();
                }
            }
            Node::Add(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Node::Num(x), Node::Num(y)) => *self = Node::Num(x + y),
                    (Node::Num(x), _) if *x == 0.0 => *self = *b.clone(),
                    (_, Node::Num(y)) if *y == 0.0 => *self = *a.clone(),
                    (Node::Neg(a_inner), Node::Neg(b_inner)) => {
                        // (-a) + (-b) = -(a + b)
                        let mut new =
                            Node::Neg(Box::new(Node::Add(a_inner.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Neg(a_inner), _) if *a_inner == *b => {
                        // (-a) + a = 0
                        *self = Node::Num(0.0);
                    }
                    (_, Node::Neg(b_inner)) if *a == *b_inner => {
                        // a + (-a) = 0
                        *self = Node::Num(0.0);
                    }
                    (Node::Neg(a_inner), _) => {
                        // (-a) + b = b - a
                        let mut new = Node::Sub(b.clone(), a_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Node::Neg(b_inner)) => {
                        // a + (-b) = a - b
                        let mut new = Node::Sub(a.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ if *a == *b => {
                        // a + a => 2*a
                        let mut new = Node::Mul(Box::new(Node::Num(2.0)), a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    // reassociate: (x + c1) + c2 / (c1 + x) + c2 => x + (c1 + c2)
                    (Node::Add(left, right), Node::Num(c2)) => {
                        if let Node::Num(c1) = right.as_ref() {
                            let mut new = Node::Add(left.clone(), Box::new(Node::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Node::Num(c1) = left.as_ref() {
                            let mut new = Node::Add(right.clone(), Box::new(Node::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    // reassociate: c1 + (x + c2) / c1 + (c2 + x) => x + (c1 + c2)
                    (Node::Num(c1), Node::Add(left, right)) => {
                        if let Node::Num(c2) = right.as_ref() {
                            let mut new = Node::Add(left.clone(), Box::new(Node::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Node::Num(c2) = left.as_ref() {
                            let mut new = Node::Add(right.clone(), Box::new(Node::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    _ => {}
                }
            }
            Node::Sub(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                if let (Node::Num(x), Node::Num(y)) = (a.as_ref(), b.as_ref()) {
                    *self = Node::Num(x - y);
                } else if let Node::Neg(b_inner) = b.as_ref() {
                    // a - (-b) => a + b
                    let mut new = Node::Add(a.clone(), b_inner.clone());
                    new.pre_eval(vars);
                    *self = new;
                } else if *a == *b {
                    // x - x => 0
                    *self = Node::Num(0.0);
                } else if let Node::Neg(a_inner) = a.as_ref() {
                    // (-a) - b = -(a + b)
                    let mut new = Node::Neg(Box::new(Node::Add(a_inner.clone(), b.clone())));
                    new.pre_eval(vars);
                    *self = new;
                } else if let Node::Num(y) = b.as_ref()
                    && *y == 0.0
                {
                    *self = *a.clone();
                } else if let Node::Num(x) = a.as_ref()
                    && *x == 0.0
                {
                    let mut new = Node::Neg(b.clone());
                    new.pre_eval(vars);
                    *self = new;
                }
            }
            Node::Mul(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Node::Num(x), Node::Num(y)) => *self = Node::Num(x * y),
                    (Node::Num(x), _) if *x == 0.0 => *self = Node::Num(0.0),
                    (_, Node::Num(y)) if *y == 0.0 => *self = Node::Num(0.0),
                    (Node::Num(x), _) if *x == 1.0 => *self = *b.clone(),
                    (_, Node::Num(y)) if *y == 1.0 => *self = *a.clone(),
                    (Node::Num(x), _) if *x == -1.0 => {
                        let mut new = Node::Neg(b.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Node::Num(y)) if *y == -1.0 => {
                        let mut new = Node::Neg(a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Neg(a_inner), Node::Neg(b_inner)) => {
                        // (-a) * (-b) = a * b
                        let mut new = Node::Mul(a_inner.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    // reassociate: (x * c1) * c2 / (c1 * x) * c2 => x * (c1 * c2)
                    (Node::Mul(left, right), Node::Num(c2)) => {
                        if let Node::Num(c1) = right.as_ref() {
                            let mut new = Node::Mul(left.clone(), Box::new(Node::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Node::Num(c1) = left.as_ref() {
                            let mut new = Node::Mul(right.clone(), Box::new(Node::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    // reassociate: c1 * (x * c2) / c1 * (c2 * x) => x * (c1 * c2)
                    (Node::Num(c1), Node::Mul(left, right)) => {
                        if let Node::Num(c2) = right.as_ref() {
                            let mut new = Node::Mul(left.clone(), Box::new(Node::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Node::Num(c2) = left.as_ref() {
                            let mut new = Node::Mul(right.clone(), Box::new(Node::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    _ => {}
                }
            }
            Node::Div(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    _ if *a == *b => {
                        // x / x => 1
                        *self = Node::Num(1.0);
                    }
                    (Node::Num(x), Node::Num(y)) => *self = Node::Num(x / y),
                    (_, Node::Num(y)) if *y == 1.0 => *self = *a.clone(),
                    (Node::Num(x), _) if *x == 0.0 => *self = Node::Num(0.0),
                    (_, Node::Num(y)) if *y == -1.0 => {
                        let mut new = Node::Neg(a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Node::Num(y)) => {
                        // x / c => x * (1/c)
                        let mut new = Node::Mul(a.clone(), Box::new(Node::Num(1.0 / *y)));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Neg(a_inner), Node::Neg(b_inner)) => {
                        // (-a) / (-b) = a / b
                        let mut new = Node::Div(a_inner.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Node::Neg(b_inner)) => {
                        // a / (-b) = -(a / b)
                        let mut new = Node::Neg(Box::new(Node::Div(a.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ => {}
                }
            }
            Node::Pow(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Node::Num(x), Node::Num(y)) => {
                        *self = Node::Num(if *x == 0.0 && *y == 0.0 {
                            1.0
                        } else {
                            x.powf(*y)
                        });
                    }
                    (_, Node::Num(y)) if *y == 0.0 => *self = Node::Num(1.0),
                    (_, Node::Num(y)) if *y == 1.0 => *self = *a.clone(),
                    (Node::Num(x), _) if *x == 1.0 => *self = Node::Num(1.0),
                    (Node::Neg(a_inner), Node::Num(y))
                        if *y == (*y as i32) as f64 && (*y as i32) % 2 == 0 =>
                    {
                        // (-x)^n => x^n  for even integer n
                        let mut new = Node::Pow(a_inner.clone(), b.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Var(_), Node::Num(y)) if *y == 2.0 => {
                        let mut new_node = Node::Mul(a.clone(), a.clone());
                        new_node.pre_eval(vars);
                        *self = new_node;
                    }
                    _ => {}
                }
            }
            Node::Mod(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Node::Num(x), Node::Num(y)) => *self = Node::Num(x % y),
                    _ if *a == *b => {
                        // x % x => 0
                        *self = Node::Num(0.0);
                    }
                    (Node::Num(x), _) if *x == 0.0 => {
                        // 0 % x => 0
                        *self = Node::Num(0.0);
                    }
                    (Node::Neg(a_inner), Node::Neg(b_inner)) => {
                        // (-a) % (-b) = -(a % b)
                        let mut new =
                            Node::Neg(Box::new(Node::Mod(a_inner.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Node::Neg(b_inner)) => {
                        // a % (-b) = a % b  (sign of divisor doesn't affect result)
                        let mut new = Node::Mod(a.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Neg(a_inner), _) => {
                        // (-a) % b = -(a % b)
                        let mut new = Node::Neg(Box::new(Node::Mod(a_inner.clone(), b.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ => {}
                }
            }
            Node::F1(f, a) => {
                a.pre_eval(vars);
                if let Node::Num(x) = a.as_ref() {
                    *self = Node::Num(f.to_fn()(*x));
                } else if let Node::F1(g, inner) = a.as_ref() {
                    match (f, g) {
                        // inverse compositions: f(g(x)) = x
                        (F1::Ln, F1::Exp) => *self = *inner.clone(),
                        (F1::Exp, F1::Ln) => *self = *inner.clone(),
                        (F1::Sin, F1::Asin) => *self = *inner.clone(),
                        (F1::Cos, F1::Acos) => *self = *inner.clone(),
                        (F1::Tan, F1::Atan) => *self = *inner.clone(),
                        // idempotent: f(f(x)) = f(x)
                        (F1::Abs, F1::Abs)
                        | (F1::Floor, F1::Floor)
                        | (F1::Ceil, F1::Ceil)
                        | (F1::Round, F1::Round)
                        | (F1::Trunc, F1::Trunc) => *self = *a.clone(),
                        _ => {}
                    }
                }
            }
            Node::F2(f, a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                if let (Node::Num(x), Node::Num(y)) = (a.as_ref(), b.as_ref()) {
                    *self = Node::Num(f.to_fn()(*x, *y));
                }
            }
            Node::Let(_slot, expr, body) => {
                expr.pre_eval(vars);
                body.pre_eval(vars);
            }
            Node::CseRef(_, _) => {}
        }
    }

    /// Return the number of CSE slots required (max slot index + 1).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node = Node::Let(0, Box::new(Node::Num(1.0)), Box::new(Node::Var(0)));
    /// assert_eq!(node.cse_slots(), 1);
    /// ```
    pub fn cse_slots(&self) -> usize {
        match self {
            Node::Let(slot, expr, body) => {
                let slot = *slot + 1;
                slot.max(expr.cse_slots()).max(body.cse_slots())
            }
            Node::CseRef(slot, _) => *slot + 1,
            Node::Neg(a) => a.cse_slots(),
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Mod(a, b)
            | Node::F2(_, a, b) => a.cse_slots().max(b.cse_slots()),
            Node::F1(_, a) => a.cse_slots(),
            Node::Num(_) | Node::Var(_) => 0,
        }
    }

    /// Apply common-subexpression elimination to the AST in-place.
    ///
    /// Repeated subtrees are extracted into `Let` bindings and evaluated once.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let mut node = Node::Mul(
    ///     Box::new(Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(1.0)))),
    ///     Box::new(Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(1.0)))),
    /// );
    /// node.cse();
    /// assert!(node.cse_slots() > 0); // (x+1) extracted into a CSE slot
    /// let mut cache = vec![0.0; node.cse_slots()];
    /// let f = node.compile();
    /// assert_eq!(f(&[4.0], &mut cache), 25.0); // (4+1)^2
    /// ```
    pub fn cse(&mut self) {
        let mut slot = 0usize;
        while self.cse_one_pass(slot) {
            slot += 1;
        }
    }

    /// Preparation pipeline: pre_eval, CSE, compile.
    /// Returns (compiled_expr, cse_slots).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::{parse, VarMap};
    /// # struct V;
    /// # impl VarMap for V {
    /// #     fn ndim(&self) -> usize { 3 }
    /// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
    /// #     fn primary_prefix(&self) -> &str { "x" }
    /// # }
    /// let mut node = parse("x*x + x*x", &V).unwrap();
    /// let (f, slots) = node.prepare(&[]);
    /// let mut cache = vec![0.0; slots];
    /// assert_eq!(f(&[3.0], &mut cache), 18.0);
    /// ```
    pub fn prepare(&mut self, vars: &[Option<f64>]) -> (CompiledExpr, usize) {
        self.pre_eval(vars);
        self.cse();
        (self.compile(), self.cse_slots())
    }

    /// Preparation pipeline: pre_eval, CSE, compile_multi.
    /// Returns compiled_expr_multi.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::{parse, VarMap};
    /// # struct V;
    /// # impl VarMap for V {
    /// #     fn ndim(&self) -> usize { 3 }
    /// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
    /// #     fn primary_prefix(&self) -> &str { "x" }
    /// # }
    /// let mut node = parse("x*y + z", &V).unwrap();
    /// let multi = node.prepare_multi(&[], &[0, 1, 2]);
    /// let mut cache = vec![0.0; multi.cse_slots];
    /// for g in &multi.groups {
    ///     (g.combined)(&[1.0, 2.0, 3.0], &mut cache);
    /// }
    /// assert_eq!((multi.main)(&[1.0, 2.0, 3.0], &mut cache), 5.0);
    /// ```
    pub fn prepare_multi(
        &mut self,
        vars: &[Option<f64>],
        spatial_dims: &[usize],
    ) -> CompiledExprMulti {
        self.pre_eval(vars);
        self.cse();
        self.compile_multi(spatial_dims)
    }

    /// Single-pass: find one repeated subtree, extract into Let.
    fn cse_one_pass(&mut self, slot: usize) -> bool {
        let pattern = {
            let mut nodes: Vec<&Node> = Vec::new();
            Self::cse_collect_extractable_candidates(self, &mut nodes);

            let mut found = None::<Node>;
            'outer: for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    if *nodes[i] == *nodes[j] {
                        found = Some(nodes[i].clone());
                        break 'outer;
                    }
                }
            }
            found
        };

        if let Some(pattern) = pattern {
            self.cse_replace_all(&pattern, slot);

            let old_self = std::mem::replace(self, Node::Num(0.0));
            *self = Node::Let(slot, Box::new(pattern), Box::new(old_self));

            true
        } else {
            false
        }
    }

    /// Collects AST nodes that can be safely extracted into a CSE `Let` binding.
    ///
    /// Returns an `IndexSet` of unbound `CseRef` slots this node depends on.
    fn cse_collect_extractable_candidates<'a>(node: &'a Node, out: &mut Vec<&'a Node>) -> IndexSet {
        match node {
            Node::CseRef(slot, _) => IndexSet::singleton(*slot),
            Node::Let(slot, expr, body) => {
                let sa = Self::cse_collect_extractable_candidates(expr, out);
                let sb = Self::cse_collect_extractable_candidates(body, out);
                let mut s = sa | sb;
                s.insert(*slot, false);
                s
            }
            Node::Num(_) | Node::Var(_) => IndexSet::default(),
            Node::Neg(a) | Node::F1(_, a) => {
                let s = Self::cse_collect_extractable_candidates(a, out);
                if s.is_empty() {
                    out.push(node);
                }
                s
            }
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Mod(a, b)
            | Node::F2(_, a, b) => {
                let sa = Self::cse_collect_extractable_candidates(a, out);
                let sb = Self::cse_collect_extractable_candidates(b, out);
                let s = sa | sb;
                if s.is_empty() {
                    out.push(node);
                }
                s
            }
        }
    }

    fn cse_replace_all(&mut self, pattern: &Node, slot: usize) {
        if *self == *pattern {
            *self = Node::CseRef(slot, pattern.depends_on());
            return;
        }
        match self {
            Node::Num(_) | Node::Var(_) | Node::CseRef(_, _) => {}
            Node::Neg(a) => a.cse_replace_all(pattern, slot),
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Mod(a, b)
            | Node::F2(_, a, b) => {
                a.cse_replace_all(pattern, slot);
                b.cse_replace_all(pattern, slot);
            }
            Node::F1(_, a) => a.cse_replace_all(pattern, slot),
            Node::Let(_, expr, body) => {
                expr.cse_replace_all(pattern, slot);
                body.cse_replace_all(pattern, slot);
            }
        }
    }

    /// Return the set of variable indices this node depends on.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node = Node::Add(Box::new(Node::Var(0)), Box::new(Node::Var(2)));
    /// let deps = node.depends_on();
    /// assert!(deps.contains(0));
    /// assert!(!deps.contains(1));
    /// assert!(deps.contains(2));
    /// ```
    pub fn depends_on(&self) -> IndexSet {
        match self {
            Node::Num(_) => IndexSet::default(),
            Node::Var(i) => IndexSet::singleton(*i),
            Node::Neg(a) | Node::F1(_, a) => a.depends_on(),
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Mod(a, b)
            | Node::F2(_, a, b) => a.depends_on() | b.depends_on(),
            Node::Let(_, expr, body) => expr.depends_on() | body.depends_on(),
            Node::CseRef(_, deps) => deps.clone(),
        }
    }

    /// Compile the AST into a closure for repeated evaluation.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node = Node::Add(Box::new(Node::Num(3.0)), Box::new(Node::Num(4.0)));
    /// let f = node.compile();
    /// assert_eq!(f(&[], &mut []), 7.0);
    /// ```
    pub fn compile(&self) -> CompiledExpr {
        match self {
            Node::Num(v) => {
                let v = *v;
                Box::new(move |_, _| v)
            }
            Node::Var(i) => {
                let i = *i;
                Box::new(move |vars: &[f64], _| vars[i])
            }
            Node::Neg(a) => {
                if let Node::Mul(x, y) = a.as_ref() {
                    let x_fn = x.compile();
                    let y_fn = y.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| {
                        -(x_fn(vars, cse) * y_fn(vars, cse))
                    })
                } else {
                    let a_fn = a.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| -a_fn(vars, cse))
                }
            }
            Node::Add(a, b) => {
                if let (Node::Mul(x, y), _) = (a.as_ref(), b.as_ref()) {
                    let x_fn = x.compile();
                    let y_fn = y.compile();
                    let c_fn = b.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| {
                        x_fn(vars, cse).mul_add(y_fn(vars, cse), c_fn(vars, cse))
                    })
                } else if let (_, Node::Mul(x, y)) = (a.as_ref(), b.as_ref()) {
                    let c_fn = a.compile();
                    let x_fn = x.compile();
                    let y_fn = y.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| {
                        x_fn(vars, cse).mul_add(y_fn(vars, cse), c_fn(vars, cse))
                    })
                } else {
                    let a_fn = a.compile();
                    let b_fn = b.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) + b_fn(vars, cse))
                }
            }
            Node::Sub(a, b) => {
                if let (Node::Mul(x, y), _) = (a.as_ref(), b.as_ref()) {
                    let x_fn = x.compile();
                    let y_fn = y.compile();
                    let c_fn = b.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| {
                        x_fn(vars, cse).mul_add(y_fn(vars, cse), -c_fn(vars, cse))
                    })
                } else if let (_, Node::Mul(x, y)) = (a.as_ref(), b.as_ref()) {
                    let c_fn = a.compile();
                    let x_fn = x.compile();
                    let y_fn = y.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| {
                        x_fn(vars, cse).mul_add(-y_fn(vars, cse), c_fn(vars, cse))
                    })
                } else {
                    let a_fn = a.compile();
                    let b_fn = b.compile();
                    Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) - b_fn(vars, cse))
                }
            }
            Node::Mul(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) * b_fn(vars, cse))
            }
            Node::Div(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) / b_fn(vars, cse))
            }
            Node::Pow(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| {
                    let base = a_fn(vars, cse);
                    let exp = b_fn(vars, cse);
                    let exp_int = exp as i32;
                    if base == 0.0 && exp == 0.0 {
                        1.0
                    } else if (exp_int as f64) == exp {
                        base.powi(exp_int)
                    } else {
                        base.powf(exp)
                    }
                })
            }
            Node::Mod(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) % b_fn(vars, cse))
            }
            Node::F1(f, a) => {
                let f = f.to_fn();
                let a_fn = a.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| f(a_fn(vars, cse)))
            }
            Node::F2(f, a, b) => {
                let f = f.to_fn();
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| f(a_fn(vars, cse), b_fn(vars, cse)))
            }
            Node::Let(slot, expr, body) => {
                let slot = *slot;
                let expr_fn = expr.compile();
                let body_fn = body.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| {
                    cse[slot] = expr_fn(vars, cse);
                    body_fn(vars, cse)
                })
            }
            Node::CseRef(slot, _) => {
                let slot = *slot;
                Box::new(move |_: &[f64], cse: &mut [f64]| cse[slot])
            }
        }
    }

    /// Extract invariant sub-expressions into pre-computed closures.
    ///
    /// `invariant_mask` indicates which variables are invariant at the
    /// current nesting level. Returns `None` if no invariants found.
    pub fn compile_invariants_combined(
        &mut self,
        invariant_mask: &IndexSet,
        slot: &mut usize,
    ) -> Option<CompiledExpr> {
        let mut pieces: Vec<(usize, Node)> = Vec::new();

        self.collect_invariants(invariant_mask, slot, &mut pieces);

        if pieces.is_empty() {
            return None;
        }

        let mut chain = Node::Num(0.0);
        for (slot, node) in pieces.into_iter().rev() {
            chain = Node::Let(slot, Box::new(node), Box::new(chain));
        }

        Some(chain.compile())
    }

    fn collect_invariants(
        &mut self,
        invariant_mask: &IndexSet,
        slot: &mut usize,
        pieces: &mut Vec<(usize, Node)>,
    ) {
        let deps = self.depends_on();
        if deps.is_disjoint(invariant_mask)
            && !matches!(self, Node::Num(_) | Node::Var(_) | Node::CseRef(_, _))
        {
            let node = std::mem::replace(self, Node::CseRef(*slot, deps));
            pieces.push((*slot, node));
            *slot += 1;
        } else {
            match self {
                Node::Num(_) | Node::Var(_) | Node::CseRef(_, _) => {}
                Node::Neg(a) | Node::F1(_, a) => a.collect_invariants(invariant_mask, slot, pieces),
                Node::Add(a, b)
                | Node::Sub(a, b)
                | Node::Mul(a, b)
                | Node::Div(a, b)
                | Node::Pow(a, b)
                | Node::Mod(a, b)
                | Node::F2(_, a, b) => {
                    a.collect_invariants(invariant_mask, slot, pieces);
                    b.collect_invariants(invariant_mask, slot, pieces);
                }
                Node::Let(ls, expr, body) => {
                    expr.collect_invariants(invariant_mask, slot, pieces);

                    let alias = if let Node::CseRef(rs, deps) = expr.as_ref()
                        && deps.is_disjoint(invariant_mask)
                        && *rs != *ls
                    {
                        Some((*ls, *rs))
                    } else {
                        None
                    };

                    if let Some((ls, rs)) = alias {
                        let mut folded = std::mem::replace(body.as_mut(), Node::Num(0.0));
                        folded.remap_cse_slot(ls, rs);
                        *self = folded;
                        self.collect_invariants(invariant_mask, slot, pieces);
                        return;
                    }

                    body.collect_invariants(invariant_mask, slot, pieces);
                }
            }
        }
    }

    fn fold_cse_aliases(&mut self) {
        if let Node::Let(slot, expr, _) = self
            && let Node::CseRef(other, _) = expr.as_ref()
            && *slot != *other
        {
            let old_slot = *slot;
            let new_slot = *other;
            if let Node::Let(_, _, body) = std::mem::replace(self, Node::Num(0.0)) {
                let mut mapped = *body;
                mapped.remap_cse_slot(old_slot, new_slot);
                mapped.fold_cse_aliases();
                *self = mapped;
            }
        } else {
            match self {
                Node::Num(_) | Node::Var(_) | Node::CseRef(_, _) => {}
                Node::Neg(a) | Node::F1(_, a) => a.fold_cse_aliases(),
                Node::Add(a, b)
                | Node::Sub(a, b)
                | Node::Mul(a, b)
                | Node::Div(a, b)
                | Node::Pow(a, b)
                | Node::Mod(a, b)
                | Node::F2(_, a, b) => {
                    a.fold_cse_aliases();
                    b.fold_cse_aliases();
                }
                Node::Let(_, expr, body) => {
                    expr.fold_cse_aliases();
                    body.fold_cse_aliases();
                }
            }
        }
    }

    fn remap_cse_slot(&mut self, old_slot: usize, new_slot: usize) {
        match self {
            Node::CseRef(slot, _) if *slot == old_slot => *slot = new_slot,
            Node::Num(_) | Node::Var(_) | Node::CseRef(_, _) => {}
            Node::Neg(a) | Node::F1(_, a) => a.remap_cse_slot(old_slot, new_slot),
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Mod(a, b)
            | Node::F2(_, a, b) => {
                a.remap_cse_slot(old_slot, new_slot);
                b.remap_cse_slot(old_slot, new_slot);
            }
            Node::Let(_, expr, body) => {
                expr.remap_cse_slot(old_slot, new_slot);
                body.remap_cse_slot(old_slot, new_slot);
            }
        }
    }

    /// Compile with multi-level invariant extraction for spatial dimensions.
    pub fn compile_multi(&mut self, spatial_dims: &[usize]) -> CompiledExprMulti {
        if spatial_dims.is_empty() {
            let main = self.compile();
            let cse_slots = self.cse_slots();
            return CompiledExprMulti {
                groups: Vec::new(),
                main,
                cse_slots,
            };
        }
        let masks: Vec<IndexSet> = {
            let rest = &spatial_dims[1..];
            let n = rest.len();
            let total = ArithIndexSet(IndexSet::singleton(n));
            let mut masks_by_popcount: Vec<Vec<IndexSet>> = vec![Vec::new(); n + 2];
            // generate only masks containing spatial_dims[0], skipping full set
            for bits in total.range_to().rev().skip(1) {
                let mut msk = IndexSet::singleton(spatial_dims[0]);
                for (i, &dim) in rest.iter().enumerate() {
                    if bits.contains(i) {
                        msk.insert(dim, true);
                    }
                }
                masks_by_popcount[msk.count_ones()].push(msk);
            }
            masks_by_popcount.into_iter().rev().flatten().collect()
        };

        let mut slot = self.cse_slots();
        let mut groups = Vec::with_capacity(masks.len());

        for mask in masks {
            if let Some(combined) = self.compile_invariants_combined(&mask, &mut slot) {
                let level = spatial_dims
                    .iter()
                    .take_while(|&&d| mask.contains(d))
                    .count();
                groups.push(InvariantGroup { level, combined });
            }
        }

        self.fold_cse_aliases();
        let main = self.compile();
        let cse_slots = self.cse_slots();

        CompiledExprMulti {
            groups,
            main,
            cse_slots,
        }
    }
}

/// Token produced by the lexer and consumed by the Pratt parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,
    Pipe,
    LParen,
    RParen,
    Comma,
    Eof,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Num(v) => write!(f, "{v}"),
            Token::Ident(s) => write!(f, "'{s}'"),
            Token::Plus => write!(f, "'+'"),
            Token::Minus => write!(f, "'-'"),
            Token::Star => write!(f, "'*'"),
            Token::Slash => write!(f, "'/'"),
            Token::Caret => write!(f, "'^'"),
            Token::Percent => write!(f, "'%'"),
            Token::Pipe => write!(f, "'|'"),
            Token::LParen => write!(f, "'('"),
            Token::RParen => write!(f, "')'"),
            Token::Comma => write!(f, "','"),
            Token::Eof => write!(f, "end of expression"),
        }
    }
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    token_start: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            token_start: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        self.pos += 1;
        c
    }

    fn next_token(&mut self) -> Result<Token, Error> {
        // skip whitespace
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }

        self.token_start = self.pos;

        if self.pos >= self.chars.len() {
            return Ok(Token::Eof);
        }

        match self.peek().expect("char must be available") {
            '+' => {
                self.advance();
                Ok(Token::Plus)
            }
            '-' => {
                self.advance();
                Ok(Token::Minus)
            }
            '*' => {
                self.advance();
                let token = if self.peek() == Some('*') {
                    self.advance();
                    Token::Caret
                } else {
                    Token::Star
                };
                Ok(token)
            }
            '/' => {
                self.advance();
                Ok(Token::Slash)
            }
            '%' => {
                self.advance();
                Ok(Token::Percent)
            }
            '|' => {
                self.advance();
                Ok(Token::Pipe)
            }
            '^' => {
                self.advance();
                Ok(Token::Caret)
            }
            '(' => {
                self.advance();
                Ok(Token::LParen)
            }
            ')' => {
                self.advance();
                Ok(Token::RParen)
            }
            ',' => {
                self.advance();
                Ok(Token::Comma)
            }
            c if c.is_ascii_digit() || c == '.' => self.read_number(),
            c if c.is_ascii_alphabetic() => self.read_ident(),
            c => Err(Error::Lexer {
                col: self.token_start + 1,
                kind: LexerErrorKind::UnexpectedChar(c),
            }),
        }
    }

    fn read_number(&mut self) -> Result<Token, Error> {
        let start = self.pos;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        if self.peek() == Some('.') {
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if let Some(c) = self.peek()
            && (c == 'e' || c == 'E')
        {
            self.advance();
            if self.peek() == Some('+') || self.peek() == Some('-') {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>().map(Token::Num).map_err(|_| Error::Lexer {
            col: self.token_start + 1,
            kind: LexerErrorKind::InvalidNumber(s),
        })
    }

    fn read_ident(&mut self) -> Result<Token, Error> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() {
                self.advance();
            } else {
                break;
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        Ok(Token::Ident(s))
    }
}

struct Parser<'a, V: VarMap> {
    lexer: Lexer,
    current: Token,
    current_pos: usize,
    vars: &'a V,
}

impl<'a, V: VarMap> Parser<'a, V> {
    fn new(input: &str, vars: &'a V) -> Result<Self, Error> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        let current_pos = lexer.token_start;
        Ok(Self {
            lexer,
            current,
            current_pos,
            vars,
        })
    }

    fn advance(&mut self) -> Result<(), Error> {
        self.current = self.lexer.next_token()?;
        self.current_pos = self.lexer.token_start;
        Ok(())
    }

    fn err_at(&self, kind: ParseErrorKind) -> Error {
        Error::Parser {
            col: self.current_pos + 1,
            kind,
        }
    }

    fn parse(mut self) -> Result<Node, Error> {
        if matches!(self.current, Token::Eof) {
            return Err(Error::Parser {
                col: 0,
                kind: ParseErrorKind::EmptyExpression,
            });
        }
        let node = self.parse_expr()?;
        if !matches!(self.current, Token::Eof) {
            return Err(self.err_at(ParseErrorKind::TrailingToken(self.current.clone())));
        }
        Ok(node)
    }

    fn parse_expr(&mut self) -> Result<Node, Error> {
        let mut left = self.parse_term()?;
        while matches!(self.current, Token::Plus | Token::Minus) {
            let op = std::mem::replace(&mut self.current, Token::Eof);
            self.advance()?;
            let right = self.parse_term()?;
            left = match op {
                Token::Plus => Node::Add(Box::new(left), Box::new(right)),
                Token::Minus => Node::Sub(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Node, Error> {
        let mut left = self.parse_unary()?;
        while matches!(self.current, Token::Star | Token::Slash | Token::Percent) {
            let op = std::mem::replace(&mut self.current, Token::Eof);
            self.advance()?;
            let right = self.parse_unary()?;
            left = match op {
                Token::Star => Node::Mul(Box::new(left), Box::new(right)),
                Token::Slash => Node::Div(Box::new(left), Box::new(right)),
                Token::Percent => Node::Mod(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Node, Error> {
        if matches!(self.current, Token::Minus) {
            self.advance()?;
            let node = self.parse_unary()?;
            Ok(Node::Neg(Box::new(node)))
        } else if matches!(self.current, Token::Plus) {
            self.advance()?;
            self.parse_unary()
        } else {
            self.parse_power()
        }
    }

    /// right-associative: a^b^c = a^(b^c)
    fn parse_power(&mut self) -> Result<Node, Error> {
        let left = self.parse_primary()?;
        if matches!(self.current, Token::Caret) {
            self.advance()?;
            let right = self.parse_power()?;
            Ok(Node::Pow(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_primary(&mut self) -> Result<Node, Error> {
        match &self.current {
            Token::Num(v) => {
                let val = *v;
                self.advance()?;
                Ok(Node::Num(val))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;
                if matches!(self.current, Token::LParen) {
                    self.parse_function_call(&name)
                } else {
                    Ok(resolve_ident(&name, self.current_pos + 1, self.vars)?)
                }
            }
            Token::LParen => {
                self.advance()?;
                let node = self.parse_expr()?;
                if !matches!(self.current, Token::RParen) {
                    return Err(self.err_at(ParseErrorKind::ExpectedRParen(self.current.clone())));
                }
                self.advance()?;
                Ok(node)
            }
            Token::Pipe => {
                self.advance()?;
                let node = self.parse_expr()?;
                if !matches!(self.current, Token::Pipe) {
                    return Err(self.err_at(ParseErrorKind::ExpectedPipe(self.current.clone())));
                }
                self.advance()?;
                Ok(Node::F1(F1::Abs, Box::new(node)))
            }
            _ => Err(self.err_at(ParseErrorKind::UnexpectedToken(self.current.clone()))),
        }
    }

    fn parse_function_call(&mut self, name: &str) -> Result<Node, Error> {
        debug_assert!(matches!(self.current, Token::LParen));
        self.advance()?;

        if matches!(self.current, Token::RParen) {
            self.advance()?;
            return match F0::from_str(name) {
                Ok(f) => Ok(Node::Num(f.to_num())),
                Err(_) => {
                    let kind = if F1::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 1,
                            found: 0,
                        }
                    } else if F2::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 2,
                            found: 0,
                        }
                    } else {
                        ParseErrorKind::UnknownIdentifier(name.to_string())
                    };
                    Err(self.err_at(kind))
                }
            };
        }

        let arg1 = self.parse_expr()?;

        if matches!(self.current, Token::Comma) {
            self.advance()?;
            let arg2 = self.parse_expr()?;
            if !matches!(self.current, Token::RParen) {
                return Err(self.err_at(ParseErrorKind::ExpectedRParen(self.current.clone())));
            }
            self.advance()?;

            match F2::from_str(name) {
                Ok(f) => Ok(Node::F2(f, Box::new(arg1), Box::new(arg2))),
                Err(_) => {
                    let kind = if F1::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 1,
                            found: 2,
                        }
                    } else if F0::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 0,
                            found: 2,
                        }
                    } else {
                        ParseErrorKind::UnknownIdentifier(name.to_string())
                    };
                    Err(self.err_at(kind))
                }
            }
        } else {
            if !matches!(self.current, Token::RParen) {
                return Err(
                    self.err_at(ParseErrorKind::ExpectedRParenOrComma(self.current.clone()))
                );
            }
            self.advance()?;

            match F1::from_str(name) {
                Ok(f) => Ok(Node::F1(f, Box::new(arg1))),
                Err(_) => {
                    let kind = if F0::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 0,
                            found: 1,
                        }
                    } else if F2::from_str(name).is_ok() {
                        ParseErrorKind::FunctionArgCount {
                            name: name.to_string(),
                            expected: 2,
                            found: 1,
                        }
                    } else {
                        ParseErrorKind::UnknownIdentifier(name.to_string())
                    };
                    Err(self.err_at(kind))
                }
            }
        }
    }
}

fn resolve_ident<V: VarMap>(name: &str, col: usize, vars: &V) -> Result<Node, Error> {
    if let Ok(f) = F0::from_str(name) {
        return Ok(Node::Num(f.to_num()));
    };

    if let Some(idx) = vars.resolve_alias(name) {
        return Ok(Node::Var(idx));
    }

    let primary = vars.primary_prefix();
    if let Some(rest) = name.strip_prefix(primary)
        && !rest.is_empty()
        && let Ok(idx) = rest.parse::<usize>()
    {
        if idx < vars.ndim() {
            return Ok(Node::Var(idx));
        }
        return Err(Error::Parser {
            col,
            kind: ParseErrorKind::VarOutOfRange {
                name: name.to_string(),
                max: vars.ndim().saturating_sub(1),
            },
        });
    }

    Err(Error::Parser {
        col,
        kind: ParseErrorKind::UnknownIdentifier(name.to_string()),
    })
}

/// Parse an expression string into a `Node` AST.
///
/// # Examples
/// ```
/// # use hypervox_expr::{parse, VarMap};
/// # struct V;
/// # impl VarMap for V {
/// #     fn ndim(&self) -> usize { 3 }
/// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
/// #     fn primary_prefix(&self) -> &str { "x" }
/// # }
/// let node = parse("x * x + y", &V).unwrap();
/// let mut cache = vec![0.0; node.cse_slots()];
/// let result = node.compile()(&[3.0, 4.0], &mut cache);
/// assert_eq!(result, 13.0);
/// ```
pub fn parse<V: VarMap>(expr_str: &str, vars: &V) -> Result<Node, Error> {
    let parser = Parser::new(expr_str, vars)?;
    parser.parse()
}

/// Validate an expression string without producing a compiled result.
///
/// # Examples
/// ```
/// # use hypervox_expr::{validate, VarMap};
/// # struct V;
/// # impl VarMap for V {
/// #     fn ndim(&self) -> usize { 3 }
/// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
/// #     fn primary_prefix(&self) -> &str { "x" }
/// # }
/// assert!(validate("x + y", &V).is_ok());
/// assert!(validate("x +", &V).is_err());
/// ```
pub fn validate(expr_str: &str, vars: &impl VarMap) -> Result<(), Error> {
    let trimmed = expr_str.trim();
    if trimmed.is_empty() {
        return Err(Error::Parser {
            col: 0,
            kind: ParseErrorKind::EmptyExpression,
        });
    }
    parse(trimmed, vars).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_eval_basic_ops() {
        let dim = D3;
        let e = |s: &str| parse(s, &dim).unwrap().compile()(&[], &mut []);
        assert_eq!(e("3 + 5"), 8.0);
        assert_eq!(e("10 - 7"), 3.0);
        assert_eq!(e("4 * 6"), 24.0);
        assert_eq!(e("15 / 3"), 5.0);
        assert_eq!(e("2 ^ 3"), 8.0);
        assert_eq!(e("--5"), 5.0);
        assert_eq!(e("(1 + 2) * 3"), 9.0);
        assert_eq!(e("2 ** 3"), 8.0);
    }

    #[test]
    fn test_eval_vars_funcs_consts() {
        let dim = D3;
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v, &mut []);
        assert_eq!(e("x + y + z", &[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(e("2 * x", &[5.0, 0.0, 0.0]), 10.0);
        assert_eq!(e("sqrt(4)", &[]), 2.0);
        assert_eq!(e("abs(-3)", &[]), 3.0);
        assert_eq!(e("atan2(1, 0)", &[]), std::f64::consts::FRAC_PI_2);
        assert_eq!(e("pow(2, 3)", &[]), 8.0);
        assert_eq!(e("pow(0, 0)", &[]), 1.0);
        assert_eq!(e("PI", &[]), std::f64::consts::PI);
        assert_eq!(e("E", &[]), std::f64::consts::E);
    }

    #[test]
    fn test_pre_eval_simplify() {
        let dim = D3;
        let pe = |s: &str, v: &[Option<f64>]| -> Node {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(v);
            n
        };

        // Full constant folding
        assert_eq!(pe("3 + 5", &[]), Node::Num(8.0));
        assert_eq!(pe("4 * 6", &[]), Node::Num(24.0));
        assert_eq!(pe("2 ^ 3", &[]), Node::Num(8.0));

        // Identity / zero elimination
        assert_eq!(pe("0 + x", &[]), Node::Var(0));
        assert_eq!(pe("x + 0", &[]), Node::Var(0));
        assert_eq!(pe("x - 0", &[]), Node::Var(0));
        assert_eq!(pe("1 * x", &[]), Node::Var(0));
        assert_eq!(pe("x * 1", &[]), Node::Var(0));
        assert_eq!(pe("x / 1", &[]), Node::Var(0));
        assert_eq!(pe("x ^ 1", &[]), Node::Var(0));
        assert_eq!(pe("0 * x", &[]), Node::Num(0.0));
        assert_eq!(pe("x * 0", &[]), Node::Num(0.0));
        assert_eq!(pe("0 / x", &[]), Node::Num(0.0));
        assert_eq!(pe("x ^ 0", &[]), Node::Num(1.0));
        assert_eq!(pe("1 ^ x", &[]), Node::Num(1.0));
        assert_eq!(pe("0 - x", &[]), Node::Neg(Box::new(Node::Var(0))));

        // Negation rewrites via -1 factor
        assert_eq!(pe("-1 * x", &[]), Node::Neg(Box::new(Node::Var(0))));
        assert_eq!(pe("x * -1", &[]), Node::Neg(Box::new(Node::Var(0))));
        assert_eq!(pe("x / -1", &[]), Node::Neg(Box::new(Node::Var(0))));

        // (-a) + (-b) = -(a + b)
        let mut n = parse("-x + -y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Neg(Box::new(Node::Add(
                Box::new(Node::Var(0)),
                Box::new(Node::Var(1)),
            )))
        );

        // a - (-b) = a + b
        let mut n = parse("x - -y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Add(Box::new(Node::Var(0)), Box::new(Node::Var(1))));

        // (-a) * (-b) = a * b
        let mut n = parse("-x * -y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Var(1))));

        // (-a) / (-b) = a / b
        let mut n = parse("-x / -y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Div(Box::new(Node::Var(0)), Box::new(Node::Var(1))));

        // ln(exp(x)) = x
        let mut n = parse("ln(exp(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Var(0));

        // idempotent: abs(abs(x)) = abs(x)
        let mut n = parse("abs(abs(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::F1(F1::Abs, Box::new(Node::Var(0))));

        // Partial substitution: x + y with x=2
        let mut n = parse("x + y", &dim).unwrap();
        n.pre_eval(&[Some(2.0), None, None]);
        assert_eq!(
            n,
            Node::Add(Box::new(Node::Num(2.0)), Box::new(Node::Var(1)))
        );

        // 0^0 = 1 through pre_eval
        assert_eq!(pe("0^0", &[]), Node::Num(1.0));
        assert_eq!(pe("pow(0, 0)", &[]), Node::Num(1.0));

        // x - x => 0
        assert_eq!(pe("x - x", &[]), Node::Num(0.0));
        assert_eq!(pe("x - x", &[None, Some(3.0), None]), Node::Num(0.0));

        // x / x => 1
        assert_eq!(pe("x / x", &[]), Node::Num(1.0));

        // a + a => 2*a
        let mut n = parse("x + x", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Mul(Box::new(Node::Num(2.0)), Box::new(Node::Var(0)))
        );

        // (-a) - b = -(a + b)
        let mut n = parse("-x - y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Neg(Box::new(Node::Add(
                Box::new(Node::Var(0)),
                Box::new(Node::Var(1)),
            )))
        );

        // a / (-b) = -(a / b)
        let mut n = parse("x / -y", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Neg(Box::new(Node::Div(
                Box::new(Node::Var(0)),
                Box::new(Node::Var(1)),
            )))
        );

        // (-x)^n => x^n for even n  (then x^2 => x*x, x^4 stays as pow)
        assert_eq!(
            pe("(-x)^2", &[]),
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Var(0)))
        );
        assert_eq!(
            pe("(-x)^4", &[]),
            Node::Pow(Box::new(Node::Var(0)), Box::new(Node::Num(4.0)))
        );
        assert_eq!(
            pe("(-x)^(-2)", &[]),
            Node::Pow(Box::new(Node::Var(0)), Box::new(Node::Num(-2.0)))
        );
        // (-x)^3 stays as Pow(Neg(x), 3) — odd n
        let mut n = parse("(-x)^3", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Pow(
                Box::new(Node::Neg(Box::new(Node::Var(0)))),
                Box::new(Node::Num(3.0))
            )
        );

        // x / c => x * (1/c)
        let mut n = parse("x / 3", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(
            n,
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(1.0 / 3.0)))
        );

        // (x + c1) + c2 => x + (c1 + c2)
        assert_eq!(
            pe("(x + 3) + 2", &[]),
            Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0)))
        );
        // (c1 + x) + c2 => x + (c1 + c2)
        assert_eq!(
            pe("(3 + x) + 2", &[]),
            Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0)))
        );
        // c1 + (x + c2) => x + (c1 + c2)
        assert_eq!(
            pe("3 + (x + 2)", &[]),
            Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0)))
        );
        // c1 + (c2 + x) => x + (c1 + c2)
        assert_eq!(
            pe("3 + (2 + x)", &[]),
            Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0)))
        );
        // deep nesting: (((x+1)+2)+3) => x + 6
        assert_eq!(
            pe("(((x + 1) + 2) + 3)", &[]),
            Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(6.0)))
        );

        // (x * c1) * c2 => x * (c1 * c2)
        assert_eq!(
            pe("(x * 3) * 2", &[]),
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0)))
        );
        // (c1 * x) * c2 => x * (c1 * c2)
        assert_eq!(
            pe("(3 * x) * 2", &[]),
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0)))
        );
        // c1 * (x * c2) => x * (c1 * c2)
        assert_eq!(
            pe("3 * (x * 2)", &[]),
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0)))
        );
        // c1 * (c2 * x) => x * (c1 * c2)
        assert_eq!(
            pe("3 * (2 * x)", &[]),
            Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0)))
        );

        // exp(ln(x)) = x
        let mut n = parse("exp(ln(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Var(0));

        // sin(asin(x)) = x
        let mut n = parse("sin(asin(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Var(0));

        // cos(acos(x)) = x
        let mut n = parse("cos(acos(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Var(0));

        // tan(atan(x)) = x
        let mut n = parse("tan(atan(x))", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::Var(0));

        // Runtime verification: all optimizations produce correct values
        let e = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&v.iter().copied().map(Some).collect::<Vec<_>>());
            n.compile()(v, &mut [])
        };
        assert_eq!(e("x - x", &[5.0, 0.0, 0.0]), 0.0);
        assert_eq!(e("x / x", &[5.0, 0.0, 0.0]), 1.0);
        assert_eq!(e("x + x", &[3.0, 0.0, 0.0]), 6.0);
        assert_eq!(e("-x - y", &[2.0, 3.0, 0.0]), -5.0);
        assert_eq!(e("x / -y", &[6.0, 2.0, 0.0]), -3.0);
        assert_eq!(e("x^(-1)", &[5.0, 0.0, 0.0]), 0.2);
        assert_eq!(e("(-x)^2", &[5.0, 0.0, 0.0]), 25.0);
        assert_eq!(e("exp(ln(x))", &[2.0, 0.0, 0.0]), 2.0);
        assert_eq!(e("sin(asin(x))", &[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(e("cos(acos(x))", &[1.0, 0.0, 0.0]), 1.0);
        assert_eq!(e("tan(atan(x))", &[0.0, 0.0, 0.0]), 0.0);
        // (-a) + a = 0  and  a + (-a) = 0
        assert_eq!(e("-x + x", &[5.0, 0.0, 0.0]), 0.0);
        assert_eq!(e("x + -x", &[5.0, 0.0, 0.0]), 0.0);
        assert_eq!(pe("-x + x", &[]), Node::Num(0.0));
        assert_eq!(pe("x + -x", &[]), Node::Num(0.0));
    }

    #[test]
    fn test_cse_basic() {
        let dim = D3;
        let cse_eval = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            let (f, slots) = n.prepare(&[]);
            let mut cache = vec![0.0; slots];
            f(v, &mut cache)
        };

        // (x+1)*(x+1)
        assert_eq!(cse_eval("(x+1)*(x+1)", &[5.0, 0.0, 0.0]), 36.0);

        // x*x + x*x
        assert_eq!(cse_eval("x*x + x*x", &[3.0, 0.0, 0.0]), 18.0);

        // sin(x)*cos(y) + sin(x)*cos(y)
        assert_eq!(
            cse_eval("sin(x)*cos(y) + sin(x)*cos(y)", &[1.0, 2.0, 0.0]),
            2.0 * (1.0_f64.sin() * 2.0_f64.cos())
        );

        let no_cse = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&[]);
            n.compile()(v, &mut [])
        };
        for &(expr, vars) in &[
            ("x*x + y*y + x*x", &[2.0, 3.0, 0.0]),
            ("x*x*x + x*x*x", &[2.0, 0.0, 0.0]),
            ("(x+y)*(x+y)", &[1.0, 2.0, 0.0]),
            ("sqrt(x*x + y*y) + sqrt(x*x + y*y)", &[3.0, 4.0, 0.0]),
        ] {
            assert_eq!(
                cse_eval(expr, vars),
                no_cse(expr, vars),
                "CSE result mismatch for '{expr}'"
            );
        }
    }

    #[test]
    fn test_cse_nested() {
        let dim = D3;
        let cse_eval = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            let (f, slots) = n.prepare(&[]);
            let mut cache = vec![0.0; slots];
            f(v, &mut cache)
        };
        let no_cse = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&[]);
            n.compile()(v, &mut [])
        };

        for &(expr, vars) in &[
            ("sin(x*x + y*y) + cos(x*x + y*y)", &[1.0, 2.0, 0.0]),
            ("sin(x*x + y*y) + cos(x*x + y*y) + x*x", &[1.0, 2.0, 0.0]),
        ] {
            assert_eq!(
                cse_eval(expr, vars),
                no_cse(expr, vars),
                "CSE nested result mismatch for '{expr}'"
            );
        }
    }

    #[test]
    fn test_cse_multiple_extractions() {
        let dim = D3;
        let cse_eval = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            let (f, slots) = n.prepare(&[]);
            let mut cache = vec![0.0; slots];
            f(v, &mut cache)
        };
        let no_cse = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&[]);
            n.compile()(v, &mut [])
        };

        for &(expr, vars) in &[
            ("x*x + x*x + y*y + y*y", &[2.0, 3.0, 0.0]),
            ("(x+1)*(x+1) + (x+1)", &[5.0, 0.0, 0.0]),
            ("x*x*y + x*x*z + x*x", &[2.0, 3.0, 4.0]),
        ] {
            assert_eq!(
                cse_eval(expr, vars),
                no_cse(expr, vars),
                "CSE multi mismatch for '{expr}'"
            );
        }
    }

    #[test]
    fn test_cse_no_duplicates() {
        let dim = D3;
        let mut n = parse("x + y + z", &dim).unwrap();
        n.pre_eval(&[]);
        n.cse();
        assert_eq!(n, parse("x + y + z", &dim).unwrap());
    }

    #[test]
    fn test_let_chain_order() {
        let pieces = vec![
            (
                1usize,
                Node::Mul(Box::new(Node::Var(2)), Box::new(Node::Var(2))),
            ),
            (0usize, Node::CseRef(1, IndexSet::singleton(2))),
            (
                2usize,
                Node::F1(
                    F1::Ln,
                    Box::new(Node::Add(
                        Box::new(Node::CseRef(1, IndexSet::singleton(2))),
                        Box::new(Node::Num(1.0)),
                    )),
                ),
            ),
        ];
        let vars = [1.5, 2.5, -3.0];

        let mut chain = Node::Num(0.0);
        for (slot, node) in pieces.clone().into_iter().rev() {
            chain = Node::Let(slot, Box::new(node), Box::new(chain));
        }
        let mut cse = vec![0.0; 10];
        let _ = chain.compile()(&vars, &mut cse);
        assert!((cse[1] - 9.0).abs() < 1e-12, "cse[1]");
        assert!(
            (cse[0] - 9.0).abs() < 1e-12,
            "cse[0] should be 9.0, got {}",
            cse[0]
        );
        assert!(
            (cse[2] - (10.0_f64).ln()).abs() < 1e-12,
            "cse[2] should be ln(10), got {}",
            cse[2]
        );
    }

    #[test]
    fn test_cse_vs_nocse() {
        let dim = D3;
        let exprs = [
            ("simple", "x + y * z"),
            ("medium", "sin(x) + cos(y) * z^2 + sqrt(x*x + y*y)"),
            (
                "heavy",
                "exp(sin(x) * cos(y)) + ln(z*z + 1) + atan2(sqrt(x*x + y*y + z*z), 1) + sqrt(abs(x + y))",
            ),
            ("repeated", "(x*x + y*y)*(x*x + y*y) + sin(x*x + y*y)"),
            (
                "very_heavy",
                concat!(
                    "exp(-sqrt(x*x + y*y + z*z) / (1 + sqrt(x*x + y*y + z*z)))",
                    " * sin(sqrt(x*x + y*y + z*z) + cos(sqrt(x*x + y*y + z*z))",
                    " * tanh(sqrt(x*x + y*y + z*z) / (1 + sqrt(x*x + y*y + z*z)))) + ",
                    "ln(1 + abs(sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
                    " + sin(x)*sin(y)*sin(z)))",
                    " * atan2(sqrt(1 + x*x + y*y + z*z",
                    " + sin(x*x + y*y + z*z)*cos(x*x + y*y + z*z)),",
                    " 1 + abs(sin(x)^2 + cos(y)^2 + tanh(z)^2",
                    " + sin(x)*sin(z) + cos(y)*cos(z))) + ",
                    "tanh(sin(x*x + y*y)*cos(y*y + z*z)*sin(z*z + x*x)",
                    " + cos(x*x + y*y)*sin(y*y + z*z)*cos(z*z + x*x))",
                    " / (1 + abs(cos(x*x + y*y + z*z)",
                    " + sin(x*x + y*y + z*z)^2)) + ",
                    "cbrt(1 + abs(sin(x*x + y*y + z*z)*cos(x*x + y*y + z*z)",
                    " - tanh((x*x + y*y + z*z) / (1 + x*x + y*y + z*z))))",
                    " * atan(sqrt(1 + x*x + y*y + z*z)",
                    " / (1 + sqrt(x*x + y*y + z*z))) + ",
                    "(sin(x)^3 + cos(y)^3 + tanh(z)^3",
                    " + sin(x)*cos(y)*tanh(z))",
                    " / (1 + abs(sin(x)*cos(y)*tanh(z)))",
                    " * sqrt(1 + sin(x)^2 + cos(y)^2 + tanh(z)^2)",
                ),
            ),
        ];
        let side = 32;
        let scale = 10.0 / side as f64;

        for (name, expr_str) in &exprs {
            let mut node1 = parse(expr_str, &dim).unwrap();
            node1.pre_eval(&[]);
            let multi1 = node1.compile_multi(&[0, 1, 2]);

            let mut node2 = parse(expr_str, &dim).unwrap();
            let multi2 = node2.prepare_multi(&[], &[0, 1, 2]);

            let mut sum1 = 0.0_f64;
            let mut sum2 = 0.0_f64;

            for nz in 0..side {
                let zv = nz as f64 * scale - 5.0;
                for ny in 0..side {
                    let yv = ny as f64 * scale - 5.0;
                    for nx in 0..side {
                        let xv = nx as f64 * scale - 5.0;
                        let vars = [xv, yv, zv];

                        let mut cache1 = vec![0.0; multi1.cse_slots];
                        for g in &multi1.groups {
                            (g.combined)(&vars, &mut cache1);
                        }
                        sum1 += (multi1.main)(&vars, &mut cache1);

                        let mut cache2 = vec![0.0; multi2.cse_slots];
                        for g in &multi2.groups {
                            (g.combined)(&vars, &mut cache2);
                        }
                        sum2 += (multi2.main)(&vars, &mut cache2);
                    }
                }
            }

            let diff = (sum1 - sum2).abs();
            let tol = 1e-12;
            assert!(
                diff < tol,
                "CSE divergence in '{name}': diff={diff:.6e} > tol={tol:.6e}"
            );
        }
    }

    #[test]
    fn test_compile_patterns_fused() {
        let dim = D3;

        let cases = [
            // MulAdd: a*b + c
            ("x*y + z", [2.0, 3.0, 4.0], 10.0),
            ("z + x*y", [2.0, 3.0, 4.0], 10.0),
            ("x*y + z", [-1.5, 2.0, 3.0], 0.0),
            ("x*y + z", [0.0, 5.0, 1.0], 1.0),
            // MulSub: a*b - c
            ("x*y - z", [2.0, 3.0, 4.0], 2.0),
            ("x*y - z", [1.0, 1.0, 0.5], 0.5),
            // NegMulAdd: c - a*b
            ("z - x*y", [2.0, 3.0, 4.0], -2.0),
            ("z - x*y", [1.0, 1.0, 5.0], 4.0),
            // NegMul: -(a*b)
            ("-(x*y)", [2.0, 3.0, 0.0], -6.0),
            ("-(x*y)", [-2.0, 3.0, 0.0], 6.0),
        ];

        for &(expr, vars, expected) in &cases {
            // Fused path (direct compile)
            let n = parse(expr, &dim).unwrap();
            let fused = n.compile()(&vars, &mut []);

            // Ref path (pre_eval + compile)
            let mut n_ref = parse(expr, &dim).unwrap();
            n_ref.pre_eval(&[]);
            let ref_val = n_ref.compile()(&vars, &mut []);

            let diff = (fused - ref_val).abs();
            assert!(
                diff < 1e-12,
                "fused mismatch for '{expr}': fused={fused}, ref={ref_val}, diff={diff:.6e}"
            );
            let exp_diff = (fused - expected).abs();
            assert!(
                exp_diff < 1e-12,
                "wrong result for '{expr}': fused={fused}, expected={expected}, diff={exp_diff:.6e}"
            );
        }

        // Fuzz: fused vs full pipeline (pre_eval + CSE + compile)
        let fuzz_cases: &[(&str, &[f64])] = &[
            ("x*y + z", &[1.5, 2.5, 3.5]),
            ("z + x*y", &[1.5, 2.5, 3.5]),
            ("x*y - z", &[1.5, 2.5, 3.5]),
            ("z - x*y", &[1.5, 2.5, 3.5]),
            ("-(x*y)", &[1.5, 2.5, 0.0]),
        ];
        for &(expr, vars) in fuzz_cases {
            let n = parse(expr, &dim).unwrap();
            let fused = n.compile()(vars, &mut []);

            let mut n_ref = parse(expr, &dim).unwrap();
            let (f_prep, slots) = n_ref.prepare(&[]);
            let mut c = vec![0.0; slots];
            let prep = f_prep(vars, &mut c);

            let diff = (fused - prep).abs();
            assert!(
                diff < 1e-12,
                "fused vs pipeline mismatch for '{expr}': fused={fused}, prep={prep}, diff={diff:.6e}"
            );
        }
    }

    #[test]
    fn test_parse_errors() {
        let dim = D3;
        assert!(parse("", &dim).is_err());
        assert!(parse("x + ", &dim).is_err());
        assert!(parse("(x + y", &dim).is_err());
        assert!(parse("unknown", &dim).is_err());
        assert!(parse("sin()", &dim).is_err());
        assert!(parse("atan2(x)", &dim).is_err());
    }

    #[test]
    fn test_percent_modulo() {
        let dim = D3;
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v, &mut []);
        assert_eq!(e("5 % 3", &[]), 2.0);
        assert_eq!(e("7 % 3.5", &[]), 0.0);
        assert_eq!(e("10 % 4", &[]), 2.0);
        assert_eq!(e("x % y", &[10.0, 3.0, 0.0]), 1.0);
        assert_eq!(e("x % 2", &[5.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn test_pipe_abs() {
        let dim = D3;
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v, &mut []);
        assert_eq!(e("|x|", &[-5.0, 0.0, 0.0]), 5.0);
        assert_eq!(e("|x|", &[3.0, 0.0, 0.0]), 3.0);
        assert_eq!(e("|x + y|", &[2.0, -3.0, 0.0]), 1.0);
        assert_eq!(e("|x| * |y|", &[-2.0, -3.0, 0.0]), 6.0);
        assert_eq!(e("-|x|", &[-5.0, 0.0, 0.0]), -5.0);
        assert_eq!(e("|x|^2", &[-3.0, 0.0, 0.0]), 9.0);
        assert_eq!(e("||x||", &[-5.0, 0.0, 0.0]), 5.0);
    }

    #[test]
    fn test_pipe_abs_pre_eval() {
        let dim = D3;
        let pe = |s: &str| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&[]);
            n
        };
        // |5| = 5
        assert_eq!(pe("|5|"), Node::Num(5.0));
        // |-3| = 3
        assert_eq!(pe("|-3|"), Node::Num(3.0));
        // abs(abs(x)) = abs(x) (idempotent)
        let mut n = parse("||x||", &dim).unwrap();
        n.pre_eval(&[]);
        assert_eq!(n, Node::F1(F1::Abs, Box::new(Node::Var(0))));
    }

    #[test]
    fn test_percent_pre_eval() {
        let dim = D3;
        let pe = |s: &str| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&[]);
            n
        };
        assert_eq!(pe("10 % 3"), Node::Num(1.0));
        // x % x => 0
        let n = pe("x % x");
        assert_eq!(n, Node::Num(0.0));
        let n = pe("(x+y) % (x+y)");
        assert_eq!(n, Node::Num(0.0));
        // 0 % x => 0
        let n = pe("0 % x");
        assert_eq!(n, Node::Num(0.0));
        // a % (-b) => a % b
        let n = pe("x % -y");
        assert_eq!(n, Node::Mod(Box::new(Node::Var(0)), Box::new(Node::Var(1))));
        // (-a) % b => -(a % b)
        let n = pe("-x % y");
        assert_eq!(
            n,
            Node::Neg(Box::new(Node::Mod(
                Box::new(Node::Var(0)),
                Box::new(Node::Var(1)),
            )))
        );
        // (-a) % (-b) => -(a % b)
        let n = pe("-x % -y");
        assert_eq!(
            n,
            Node::Neg(Box::new(Node::Mod(
                Box::new(Node::Var(0)),
                Box::new(Node::Var(1)),
            )))
        );
    }

    #[test]
    fn test_percent_runtime_optimizations() {
        let dim = D3;
        let e = |s: &str, v: &[f64]| {
            let mut n = parse(s, &dim).unwrap();
            n.pre_eval(&v.iter().copied().map(Some).collect::<Vec<_>>());
            n.compile()(v, &mut [])
        };
        // x % x = 0
        assert_eq!(e("x % x", &[5.0, 0.0, 0.0]), 0.0);
        // 0 % x = 0
        assert_eq!(e("0 % x", &[5.0, 0.0, 0.0]), 0.0);
        // (-a) % b = -(a % b)
        assert_eq!(e("-x % y", &[10.0, 3.0, 0.0]), -(10.0 % 3.0));
        // a % (-b) = a % b
        assert_eq!(e("x % -y", &[10.0, 3.0, 0.0]), 10.0 % 3.0);
        // (-a) % (-b) = -(a % b)
        assert_eq!(e("-x % -y", &[10.0, 3.0, 0.0]), -(10.0 % 3.0));
    }

    #[test]
    fn test_nd_variables() {
        let dim = TestVars { ndim: 4 };
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v, &mut []);
        assert_eq!(e("x0", &[5.0, 0.0, 0.0, 0.0]), 5.0);
        assert_eq!(e("x3", &[0.0, 0.0, 0.0, 5.0]), 5.0);
        assert!(parse("x4", &dim).is_err());
        assert!(parse("xabc", &dim).is_err());
    }

    #[test]
    fn test_error_structure() {
        let dim = D3;
        let err = parse("(x + y", &dim).unwrap_err();
        assert!(matches!(
            err,
            Error::Parser {
                col: _,
                kind: ParseErrorKind::ExpectedRParen(_)
            }
        ));
    }

    #[test]
    fn test_compile_multi_10_dim() {
        let dim = TestVars { ndim: 10 };
        let expr = "x0*x0 + x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5 + x6*x6 + x7*x7 + x8*x8 + x9*x9";
        let vars = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let expected: f64 = vars.iter().map(|v| v * v).sum();
        let tol = 1e-12;

        // compile_multi with 10 spatial dims
        let mut node = parse(expr, &dim).unwrap();
        node.pre_eval(&[]);
        let multi = node.compile_multi(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut cache = vec![0.0; multi.cse_slots];
        for g in &multi.groups {
            (g.combined)(&vars, &mut cache);
        }
        let result = (multi.main)(&vars, &mut cache);

        // Reference: direct compile (no multi)
        let node_ref = parse(expr, &dim).unwrap();
        let reference = node_ref.compile()(&vars, &mut []);

        assert!(
            (result - reference).abs() < tol,
            "compile_multi(10-dim)={result} != reference={reference}"
        );
        assert!(
            (result - expected).abs() < tol,
            "compile_multi(10-dim)={result} != expected={expected}"
        );
    }
}
