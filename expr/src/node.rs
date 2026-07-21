use crate::index_set::{ArithIndexSet, IndexSet};
use crate::{F1, F2};

/// A compiled expression closure: `(vars, cse_cache) -> result`.
pub type CompiledExpr = Box<dyn Fn(&[f64], &mut [f64]) -> f64 + Send + Sync>;

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
