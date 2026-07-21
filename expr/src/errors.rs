use std::fmt::{self, Display};

use crate::LexerToken;

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
    TrailingToken(LexerToken),
    ExpectedRParen(LexerToken),
    ExpectedPipe(LexerToken),
    UnexpectedToken(LexerToken),
    FunctionArgCount {
        name: String,
        expected: usize,
        found: usize,
    },
    ExpectedRParenOrComma(LexerToken),
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

/// Error type for `TryFrom` conversions involving `ArithIndexSet`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithIndexSetTryFromError {
    /// The source value exceeds the target type's range.
    Overflow,
    /// The source value is negative (only for signed-to-`ArithIndexSet`).
    Negative,
}

impl std::fmt::Display for ArithIndexSetTryFromError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArithIndexSetTryFromError::Overflow => {
                write!(f, "value too large for target type")
            }
            ArithIndexSetTryFromError::Negative => {
                write!(f, "negative value cannot be represented as ArithIndexSet")
            }
        }
    }
}

impl std::error::Error for ArithIndexSetTryFromError {}
