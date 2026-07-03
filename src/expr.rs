use std::{
    fmt::Display,
    ops::{BitAnd, BitOr},
    str::FromStr,
};

use crate::math::DimConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepMask(u128);

impl From<u128> for DepMask {
    fn from(value: u128) -> Self {
        DepMask(value)
    }
}

impl BitOr for DepMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Output::from(self.0 | rhs.0)
    }
}

impl BitAnd for DepMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Output::from(self.0 & rhs.0)
    }
}

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
    F1(F1, Box<Node>),
    F2(F2, Box<Node>, Box<Node>),
    /// let slot_i = expr in body
    Let(usize, Box<Node>, Box<Node>),
    /// reference to cached CSE slot
    CseRef(usize, DepMask),
}

pub type CompiledExpr = Box<dyn Fn(&[f64], &mut [f64]) -> f64 + Send + Sync>;

#[derive(Debug, Clone, Copy)]
pub enum F0 {
    PI,
    E,
}

impl F0 {
    pub fn to_num(self) -> f64 {
        match self {
            Self::PI => std::f64::consts::PI,
            Self::E => std::f64::consts::E,
        }
    }
}

impl FromStr for F0 {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "PI" => Ok(Self::PI),
            "E" => Ok(Self::E),
            _ => Err(format!("unknown const '{s}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F1 {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Sqrt,
    Cbrt,
    Exp,
    Ln,
    Log10,
    Log2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Abs,
}

impl FromStr for F1 {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sin" => Ok(Self::Sin),
            "cos" => Ok(Self::Cos),
            "tan" => Ok(Self::Tan),
            "asin" => Ok(Self::Asin),
            "acos" => Ok(Self::Acos),
            "atan" => Ok(Self::Atan),
            "sinh" => Ok(Self::Sinh),
            "cosh" => Ok(Self::Cosh),
            "tanh" => Ok(Self::Tanh),
            "sqrt" => Ok(Self::Sqrt),
            "cbrt" => Ok(Self::Cbrt),
            "exp" => Ok(Self::Exp),
            "ln" => Ok(Self::Ln),
            "log10" => Ok(Self::Log10),
            "log2" => Ok(Self::Log2),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            "round" => Ok(Self::Round),
            "trunc" => Ok(Self::Trunc),
            "abs" => Ok(Self::Abs),
            _ => Err(format!("unknown function '{s}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F2 {
    Atan2,
    Pow,
}

impl FromStr for F2 {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "atan2" => Ok(Self::Atan2),
            "pow" => Ok(Self::Pow),
            _ => Err(format!("unknown function '{s}'")),
        }
    }
}

pub struct InvariantGroup {
    pub level: u8,
    pub combined: CompiledExpr,
}

pub struct CompiledExprMulti {
    pub groups: Vec<InvariantGroup>,
    pub main: CompiledExpr,
    pub cse_slots: usize,
}

impl Node {
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
            Node::F1(f, a) => {
                a.pre_eval(vars);
                if let Node::Num(x) = a.as_ref() {
                    *self = Node::Num(match f {
                        F1::Sin => x.sin(),
                        F1::Cos => x.cos(),
                        F1::Tan => x.tan(),
                        F1::Asin => x.asin(),
                        F1::Acos => x.acos(),
                        F1::Atan => x.atan(),
                        F1::Sinh => x.sinh(),
                        F1::Cosh => x.cosh(),
                        F1::Tanh => x.tanh(),
                        F1::Sqrt => x.sqrt(),
                        F1::Cbrt => x.cbrt(),
                        F1::Exp => x.exp(),
                        F1::Ln => x.ln(),
                        F1::Log10 => x.log10(),
                        F1::Log2 => x.log2(),
                        F1::Floor => x.floor(),
                        F1::Ceil => x.ceil(),
                        F1::Round => x.round(),
                        F1::Trunc => x.trunc(),
                        F1::Abs => x.abs(),
                    });
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
                    *self = Node::Num(match f {
                        F2::Atan2 => x.atan2(*y),
                        F2::Pow => {
                            if *x == 0.0 && *y == 0.0 {
                                1.0
                            } else {
                                x.powf(*y)
                            }
                        }
                    });
                }
            }
            Node::Let(_slot, expr, body) => {
                expr.pre_eval(vars);
                body.pre_eval(vars);
            }
            Node::CseRef(_, _) => {}
        }
    }

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
            | Node::F2(_, a, b) => a.cse_slots().max(b.cse_slots()),
            Node::F1(_, a) => a.cse_slots(),
            Node::Num(_) | Node::Var(_) => 0,
        }
    }

    pub fn cse(&mut self) {
        let mut slot = 0usize;
        while self.cse_one_pass(slot) {
            slot += 1;
        }
    }

    /// Preparation pipeline: pre_eval, CSE, compile.
    /// Returns (compiled_expr, cse_slots).
    #[cfg(test)]
    pub fn prepare(&mut self, vars: &[Option<f64>]) -> (CompiledExpr, usize) {
        self.pre_eval(vars);
        self.cse();
        (self.compile(), self.cse_slots())
    }

    /// Preparation pipeline: pre_eval, CSE, compile_multi.
    /// Returns compiled_expr_multi.
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
            Self::cse_collect_non_trivial(self, &mut nodes);

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

    fn cse_collect_non_trivial<'a>(node: &'a Node, out: &mut Vec<&'a Node>) {
        match node {
            Node::Num(_) | Node::Var(_) | Node::CseRef(_, _) => {}
            Node::Let(_, expr, body) => {
                Self::cse_collect_non_trivial(expr, out);
                Self::cse_collect_non_trivial(body, out);
            }
            Node::Neg(a) => {
                out.push(node);
                Self::cse_collect_non_trivial(a, out);
            }
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::F2(_, a, b) => {
                out.push(node);
                Self::cse_collect_non_trivial(a, out);
                Self::cse_collect_non_trivial(b, out);
            }
            Node::F1(_, a) => {
                out.push(node);
                Self::cse_collect_non_trivial(a, out);
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

    pub fn depends_on(&self) -> DepMask {
        match self {
            Node::Num(_) => 0.into(),
            Node::Var(i) => (1 << *i).into(),
            Node::Neg(a) | Node::F1(_, a) => a.depends_on(),
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::F2(_, a, b) => a.depends_on() | b.depends_on(),
            Node::Let(_, expr, body) => expr.depends_on() | body.depends_on(),
            Node::CseRef(_, deps) => *deps,
        }
    }

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
                let a_fn = a.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| -a_fn(vars, cse))
            }
            Node::Add(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) + b_fn(vars, cse))
            }
            Node::Sub(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| a_fn(vars, cse) - b_fn(vars, cse))
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
            Node::F1(f, a) => {
                let f = *f;
                let a_fn = a.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| {
                    let v = a_fn(vars, cse);
                    match f {
                        F1::Sin => v.sin(),
                        F1::Cos => v.cos(),
                        F1::Tan => v.tan(),
                        F1::Asin => v.asin(),
                        F1::Acos => v.acos(),
                        F1::Atan => v.atan(),
                        F1::Sinh => v.sinh(),
                        F1::Cosh => v.cosh(),
                        F1::Tanh => v.tanh(),
                        F1::Sqrt => v.sqrt(),
                        F1::Cbrt => v.cbrt(),
                        F1::Exp => v.exp(),
                        F1::Ln => v.ln(),
                        F1::Log10 => v.log10(),
                        F1::Log2 => v.log2(),
                        F1::Floor => v.floor(),
                        F1::Ceil => v.ceil(),
                        F1::Round => v.round(),
                        F1::Trunc => v.trunc(),
                        F1::Abs => v.abs(),
                    }
                })
            }
            Node::F2(f, a, b) => {
                let f = *f;
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| match f {
                    F2::Atan2 => a_fn(vars, cse).atan2(b_fn(vars, cse)),
                    F2::Pow => {
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
                    }
                })
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

    pub fn compile_invariants_combined(
        &mut self,
        invariant_mask: DepMask,
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
        invariant_mask: DepMask,
        slot: &mut usize,
        pieces: &mut Vec<(usize, Node)>,
    ) {
        let deps = self.depends_on();
        if deps & invariant_mask == 0.into()
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
                | Node::F2(_, a, b) => {
                    a.collect_invariants(invariant_mask, slot, pieces);
                    b.collect_invariants(invariant_mask, slot, pieces);
                }
                Node::Let(ls, expr, body) => {
                    expr.collect_invariants(invariant_mask, slot, pieces);

                    let alias = if let Node::CseRef(rs, deps) = expr.as_ref()
                        && *deps & invariant_mask == 0.into()
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

    pub fn compile_multi(&mut self, spatial_dims: &[usize]) -> CompiledExprMulti {
        let masks: Vec<DepMask> = {
            let n = spatial_dims.len();
            let total = 1u8 << n;
            let mut masks_by_popcount: Vec<Vec<DepMask>> = vec![Vec::new(); n + 1];

            for bits in 1..(total - 1) {
                let msk = (0..n)
                    .filter(|&i| (bits >> i) & 1 == 1)
                    .map(|i| 1u128 << spatial_dims[i])
                    .sum::<u128>()
                    .into();
                masks_by_popcount[bits.count_ones() as usize].push(msk);
            }

            masks_by_popcount.into_iter().rev().flatten().collect()
        };

        let inner_bit: DepMask = (1u128 << spatial_dims[0]).into();

        let mut slot = self.cse_slots();
        let mut groups = Vec::with_capacity(masks.len());

        for mask in masks {
            // Skip masks without x (innermost) — nothing to hoist to
            if mask & inner_bit == 0.into() {
                continue;
            }
            if let Some(combined) = self.compile_invariants_combined(mask, &mut slot) {
                let level = spatial_dims
                    .iter()
                    .take_while(|&&d| (mask & DepMask::from(1u128 << d)) != 0.into())
                    .count() as u8;
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

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
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

    fn next_token(&mut self) -> Result<Token, String> {
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
            c => Err(format!(
                "at column {}: unexpected character '{c}'",
                self.token_start + 1
            )),
        }
    }

    fn read_number(&mut self) -> Result<Token, String> {
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
        s.parse::<f64>()
            .map(Token::Num)
            .map_err(|_| format!("at column {}: invalid number '{s}'", self.token_start + 1))
    }

    fn read_ident(&mut self) -> Result<Token, String> {
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

struct Parser<'a> {
    lexer: Lexer,
    current: Token,
    current_pos: usize,
    dim_config: &'a DimConfig,
}

impl<'a> Parser<'a> {
    fn new(input: &str, dim_config: &'a DimConfig) -> Result<Self, String> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        let current_pos = lexer.token_start;
        Ok(Self {
            lexer,
            current,
            current_pos,
            dim_config,
        })
    }

    fn advance(&mut self) -> Result<(), String> {
        self.current = self.lexer.next_token()?;
        self.current_pos = self.lexer.token_start;
        Ok(())
    }

    fn err_at(&self, msg: impl Display) -> String {
        format!("at column {}: {msg}", self.current_pos + 1)
    }

    fn parse(mut self) -> Result<Node, String> {
        if matches!(self.current, Token::Eof) {
            return Err("expression cannot be empty".into());
        }
        let node = self.parse_expr()?;
        if !matches!(self.current, Token::Eof) {
            return Err(self.err_at(format!(
                "unexpected token {} after expression",
                self.current
            )));
        }
        Ok(node)
    }

    fn parse_expr(&mut self) -> Result<Node, String> {
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

    fn parse_term(&mut self) -> Result<Node, String> {
        let mut left = self.parse_unary()?;
        while matches!(self.current, Token::Star | Token::Slash) {
            let op = std::mem::replace(&mut self.current, Token::Eof);
            self.advance()?;
            let right = self.parse_unary()?;
            left = match op {
                Token::Star => Node::Mul(Box::new(left), Box::new(right)),
                Token::Slash => Node::Div(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Node, String> {
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
    fn parse_power(&mut self) -> Result<Node, String> {
        let left = self.parse_primary()?;
        if matches!(self.current, Token::Caret) {
            self.advance()?;
            let right = self.parse_power()?;
            Ok(Node::Pow(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_primary(&mut self) -> Result<Node, String> {
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
                    Ok(resolve_ident(&name, self.current_pos + 1, self.dim_config)?)
                }
            }
            Token::LParen => {
                self.advance()?;
                let node = self.parse_expr()?;
                if !matches!(self.current, Token::RParen) {
                    return Err(self.err_at(format!("expected ')' but found {}", self.current)));
                }
                self.advance()?;
                Ok(node)
            }
            _ => Err(self.err_at(format!("unexpected token {}", self.current))),
        }
    }

    fn parse_function_call(&mut self, name: &str) -> Result<Node, String> {
        debug_assert!(matches!(self.current, Token::LParen));
        self.advance()?;

        if matches!(self.current, Token::RParen) {
            self.advance()?;
            return match F0::from_str(name) {
                Ok(f) => Ok(Node::Num(f.to_num())),
                Err(e) => Err(self.err_at(if F1::from_str(name).is_ok() {
                    format!("function '{name}' requires 1 argument")
                } else if F2::from_str(name).is_ok() {
                    format!("function '{name}' requires 2 arguments")
                } else {
                    e
                })),
            };
        }

        let arg1 = self.parse_expr()?;

        if matches!(self.current, Token::Comma) {
            self.advance()?;
            let arg2 = self.parse_expr()?;
            if !matches!(self.current, Token::RParen) {
                return Err(self.err_at(format!("expected ')' but found {}", self.current)));
            }
            self.advance()?;

            match F2::from_str(name) {
                Ok(f) => Ok(Node::F2(f, Box::new(arg1), Box::new(arg2))),
                Err(e) => Err(self.err_at(if F1::from_str(name).is_ok() {
                    format!("function '{name}' requires 1 argument")
                } else if F0::from_str(name).is_ok() {
                    format!("function '{name}' requires 0 arguments")
                } else {
                    e
                })),
            }
        } else {
            if !matches!(self.current, Token::RParen) {
                return Err(self.err_at(format!("expected ')' or ',' but found {}", self.current)));
            }
            self.advance()?;

            match F1::from_str(name) {
                Ok(f) => Ok(Node::F1(f, Box::new(arg1))),
                Err(e) => Err(self.err_at(if F0::from_str(name).is_ok() {
                    format!("function '{name}' requires 0 arguments")
                } else if F2::from_str(name).is_ok() {
                    format!("function '{name}' requires 2 arguments")
                } else {
                    e
                })),
            }
        }
    }
}

fn resolve_ident(name: &str, col: usize, dim_config: &DimConfig) -> Result<Node, String> {
    if let Ok(f) = F0::from_str(name) {
        return Ok(Node::Num(f.to_num()));
    };

    let err_prefix = format!("at column {col}");

    match name {
        "x" => Ok(Node::Var(dim_config.x_dim)),
        "y" => Ok(Node::Var(dim_config.y_dim)),
        "z" => Ok(Node::Var(dim_config.z_dim)),
        s if s.starts_with('x') && s.len() > 1 => {
            let rest = &s[1..];
            let idx = rest.parse::<usize>().map_err(|_| {
                if rest.chars().all(|c| c.is_ascii_digit()) {
                    format!(
                        "{}: variable '{s}' out of range: max index is {}",
                        err_prefix,
                        dim_config.ndim - 1
                    )
                } else {
                    format!("{}: unknown identifier '{s}'", err_prefix)
                }
            })?;
            if idx >= dim_config.ndim {
                Err(format!(
                    "{}: variable '{s}' out of range: max index is {}",
                    err_prefix,
                    dim_config.ndim - 1
                ))
            } else {
                Ok(Node::Var(idx))
            }
        }
        s => Err(format!("{}: unknown identifier '{s}'", err_prefix)),
    }
}

/// Parse an expression string into a `Node` AST.
pub fn parse(expr_str: &str, dim_config: &DimConfig) -> Result<Node, String> {
    let parser = Parser::new(expr_str, dim_config)?;
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DimConfig;

    #[test]
    fn test_eval_basic_ops() {
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
        let dim = DimConfig::default();
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
            (0usize, Node::CseRef(1, DepMask::from(0))),
            (
                2usize,
                Node::F1(
                    F1::Ln,
                    Box::new(Node::Add(
                        Box::new(Node::CseRef(0, DepMask::from(4))),
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
        let dim = DimConfig::default();
        let exprs = [
            ("simple", "x + y * z"),
            ("medium", "sin(x) + cos(y) * z^2 + sqrt(x*x + y*y)"),
            (
                "heavy",
                "exp(sin(x) * cos(y)) + ln(z*z + 1) + atan2(sqrt(x*x + y*y + z*z), 1) + sqrt(abs(x + y))",
            ),
            ("repeated", "(x*x + y*y)*(x*x + y*y) + sin(x*x + y*y)"),
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
    fn test_parse_errors() {
        let dim = DimConfig::default();
        assert!(parse("", &dim).is_err());
        assert!(parse("x + ", &dim).is_err());
        assert!(parse("(x + y", &dim).is_err());
        assert!(parse("unknown", &dim).is_err());
        assert!(parse("sin()", &dim).is_err());
        assert!(parse("atan2(x)", &dim).is_err());
    }

    #[test]
    fn test_nd_variables() {
        let dim = DimConfig {
            ndim: 4,
            x_dim: 1,
            y_dim: 2,
            z_dim: 3,
            ..DimConfig::default()
        };
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v, &mut []);
        assert_eq!(e("x0", &[5.0, 0.0, 0.0, 0.0]), 5.0);
        assert_eq!(e("x3", &[0.0, 0.0, 0.0, 5.0]), 5.0);
        assert!(parse("x4", &dim).is_err());
        assert!(parse("xabc", &dim).is_err());
    }
}
