use std::{fmt::Display, str::FromStr};

use crate::math::DimConfig;

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
}

pub type CompiledExpr = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

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
                    _ => {}
                }
            }
            Node::Div(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Node::Num(x), Node::Num(y)) => *self = Node::Num(x / y),
                    (_, Node::Num(y)) if *y == 1.0 => *self = *a.clone(),
                    (Node::Num(x), _) if *x == 0.0 => *self = Node::Num(0.0),
                    (_, Node::Num(y)) if *y == -1.0 => {
                        let mut new = Node::Neg(a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Node::Neg(a_inner), Node::Neg(b_inner)) => {
                        // (-a) / (-b) = a / b
                        let mut new = Node::Div(a_inner.clone(), b_inner.clone());
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
                        // ln(exp(x)) = x
                        (F1::Ln, F1::Exp) => *self = *inner.clone(),
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
        }
    }

    pub fn compile(&self) -> CompiledExpr {
        match self {
            Node::Num(v) => {
                let v = *v;
                Box::new(move |_| v)
            }
            Node::Var(i) => {
                let i = *i;
                Box::new(move |vars: &[f64]| vars[i])
            }
            Node::Neg(a) => {
                let a_fn = a.compile();
                Box::new(move |vars: &[f64]| -a_fn(vars))
            }
            Node::Add(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64]| a_fn(vars) + b_fn(vars))
            }
            Node::Sub(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64]| a_fn(vars) - b_fn(vars))
            }
            Node::Mul(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64]| a_fn(vars) * b_fn(vars))
            }
            Node::Div(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64]| a_fn(vars) / b_fn(vars))
            }
            Node::Pow(a, b) => {
                let a_fn = a.compile();
                let b_fn = b.compile();
                Box::new(move |vars: &[f64]| {
                    let base = a_fn(vars);
                    let exp = b_fn(vars);
                    if base == 0.0 && exp == 0.0 {
                        1.0
                    } else {
                        base.powf(exp)
                    }
                })
            }
            Node::F1(f, a) => {
                let f = *f;
                let a_fn = a.compile();
                Box::new(move |vars: &[f64]| {
                    let v = a_fn(vars);
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
                Box::new(move |vars: &[f64]| match f {
                    F2::Atan2 => a_fn(vars).atan2(b_fn(vars)),
                    F2::Pow => {
                        let base = a_fn(vars);
                        let exp = b_fn(vars);
                        if base == 0.0 && exp == 0.0 {
                            1.0
                        } else {
                            base.powf(exp)
                        }
                    }
                })
            }
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
        let e = |s: &str| parse(s, &dim).unwrap().compile()(&[]);
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
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v);
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
        let e = |s: &str, v: &[f64]| parse(s, &dim).unwrap().compile()(v);
        assert_eq!(e("x0", &[5.0, 0.0, 0.0, 0.0]), 5.0);
        assert_eq!(e("x3", &[0.0, 0.0, 0.0, 5.0]), 5.0);
        assert!(parse("x4", &dim).is_err());
        assert!(parse("xabc", &dim).is_err());
    }
}
