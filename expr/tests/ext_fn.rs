use hypervox_expr::*;
use std::str::FromStr;

// Empty enums

define_ext_f0!(EmptyF0);
define_ext_f1!(EmptyF1);
define_ext_f2!(EmptyF2);

#[test]
fn test_empty_f0() {
    assert!(EmptyF0::from_str("anything").is_err());
}

#[test]
fn test_empty_f1() {
    assert!(EmptyF1::from_str("anything").is_err());
}

#[test]
fn test_empty_f2() {
    assert!(EmptyF2::from_str("anything").is_err());
}

// TestF0

define_ext_f0!(
    TestF0,
    Zero => "zero" = 0f64,
    One => "one" = 1f64,
);

#[test]
fn test_f0_from_str() {
    assert_eq!(TestF0::from_str("zero").unwrap(), TestF0::Zero);
    assert_eq!(TestF0::from_str("one").unwrap(), TestF0::One);
    assert!(TestF0::from_str("unknown").is_err());
}

#[test]
fn test_f0_to_num() {
    assert_eq!(TestF0::Zero.to_num(), 0.0);
    assert_eq!(TestF0::One.to_num(), 1.0);
}

#[test]
fn test_f0_names() {
    assert_eq!(TestF0::Zero.names(), &["zero", "one"]);
    assert_eq!(TestF0::One.names(), &["zero", "one"]);
}

#[test]
fn test_f0_list() {
    let list = TestF0::Zero.list();
    assert_eq!(list, "zero, one");
    assert!(list.contains("zero"));
    assert!(list.contains("one"));
}

// TestF1

define_ext_f1!(
    TestF1,
    Double => "double" = |x| x * 2f64,
    Square => "square" = |x| x * x,
);

#[test]
fn test_f1_from_str() {
    assert_eq!(TestF1::from_str("double").unwrap(), TestF1::Double);
    assert_eq!(TestF1::from_str("square").unwrap(), TestF1::Square);
    assert!(TestF1::from_str("unknown").is_err());
}

#[test]
fn test_f1_to_fn() {
    let f = TestF1::Double.to_fn();
    assert_eq!(f(0.0), 0.0);
    assert_eq!(f(1.0), 2.0);
    assert_eq!(f(-3.0), -6.0);

    let g = TestF1::Square.to_fn();
    assert_eq!(g(0.0), 0.0);
    assert_eq!(g(3.0), 9.0);
    assert_eq!(g(-4.0), 16.0);
}

#[test]
fn test_f1_names() {
    assert_eq!(TestF1::Double.names(), &["double", "square"]);
}

#[test]
fn test_f1_list() {
    assert_eq!(TestF1::Double.list(), "double, square");
}

// TestF2

define_ext_f2!(
    TestF2,
    Sum => "sum" = |x, y| x + y,
    Product => "product" = |x, y| x * y,
);

#[test]
fn test_f2_from_str() {
    assert_eq!(TestF2::from_str("sum").unwrap(), TestF2::Sum);
    assert_eq!(TestF2::from_str("product").unwrap(), TestF2::Product);
    assert!(TestF2::from_str("unknown").is_err());
}

#[test]
fn test_f2_to_fn() {
    let f = TestF2::Sum.to_fn();
    assert_eq!(f(0.0, 0.0), 0.0);
    assert_eq!(f(1.0, 2.0), 3.0);
    assert_eq!(f(-1.0, 1.0), 0.0);

    let g = TestF2::Product.to_fn();
    assert_eq!(g(0.0, 5.0), 0.0);
    assert_eq!(g(2.0, 3.0), 6.0);
    assert_eq!(g(-2.0, 4.0), -8.0);
}

#[test]
fn test_f2_names() {
    assert_eq!(TestF2::Sum.names(), &["sum", "product"]);
}

#[test]
fn test_f2_list() {
    assert_eq!(TestF2::Sum.list(), "sum, product");
}

// Integration: parse_with_ext + compile

#[derive(Clone, Copy)]
struct TestVars;

impl VarMap for TestVars {
    fn ndim(&self) -> usize {
        3
    }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name {
            "x" => Some(0),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        }
    }
    fn primary_prefix(&self) -> &str {
        "x"
    }
}

define_ext_f0!(MyF0, Answer => "answer" = 42.0);

define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);

define_ext_f2!(MyF2, Hypot => "hypot" = |x, y| x.hypot(y));

#[test]
fn test_ext_f0_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("answer", &TestVars).unwrap();
    assert_eq!(node.compile()(&[], &mut []), 42.0);
}

#[test]
fn test_ext_f1_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("cube(x)", &TestVars).unwrap();
    assert_eq!(node.compile()(&[3.0, 0.0, 0.0], &mut []), 27.0);
}

#[test]
fn test_ext_f2_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("hypot(3, 4)", &TestVars).unwrap();
    assert_eq!(node.compile()(&[], &mut []), 5.0);
}

#[test]
fn test_ext_combined_expression() {
    let node =
        parse_with_ext::<_, MyF0, MyF1, MyF2>("answer + cube(x) + hypot(y, z)", &TestVars).unwrap();
    let result = node.compile()(&[2.0, 3.0, 4.0], &mut []);
    // 42 + 2^3 + sqrt(3^2 + 4^2) = 42 + 8 + 5 = 55
    assert_eq!(result, 55.0);
}

#[test]
fn test_ext_nested_expression() {
    let node =
        parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(answer + x) - hypot(y, z)", &TestVars).unwrap();
    let result = node.compile()(&[1.0, 3.0, 4.0], &mut []);
    // cube(42 + 1) - sqrt(3^2 + 4^2) = 43^3 - 5 = 79507 - 5 = 79502
    assert_eq!(result, 79507.0 - 5.0);
}

#[test]
fn test_ext_f1_in_pre_eval() {
    let mut node = parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(2)", &TestVars).unwrap();
    node.pre_eval(&[]);
    // 2^3 = 8, should fold to constant
    assert_eq!(node, Node::Num(8.0))
}

#[test]
fn test_ext_validate() {
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("answer + x", &TestVars).is_ok());
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("unknown", &TestVars).is_err());
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("", &TestVars).is_err());
}

#[test]
fn test_ext_parse_error() {
    let err = parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(1", &TestVars).unwrap_err();
    assert!(matches!(
        err,
        Error::Parser {
            kind: ParseErrorKind::ExpectedRParenOrComma(_),
            ..
        }
    ));
}
