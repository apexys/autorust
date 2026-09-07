//! Minimal port of tinygrad's env-var plumbing (`helpers.py` getenv + ContextVar).
//!
//! tinygrad's `Context` pushes a dict of overrides onto a stack; `getenv` reads the
//! innermost override first, then the process environment, then the default.
//! We keep the same semantics with a thread-local override stack.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static OVERRIDES: RefCell<Vec<HashMap<String, String>>> = const { RefCell::new(Vec::new()) };
}

/// Push a context scope with overrides (key -> value as string).
pub fn push_context(overrides: &[(&str, String)]) {
    let map: HashMap<String, String> =
        overrides.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    OVERRIDES.with_borrow_mut(|s| s.push(map));
}

/// Pop the innermost context scope. No-op on empty stack (mirrors `Context.__exit__` tolerance).
pub fn pop_context() {
    OVERRIDES.with_borrow_mut(|s| {
        s.pop();
    });
}

/// Run `f` with the given context overrides active (tinygrad's `with Context(...)`) scope.
pub fn with_context<T>(overrides: &[(&str, String)], f: impl FnOnce() -> T) -> T {
    push_context(overrides);
    let out = f();
    pop_context();
    out
}

fn lookup(name: &str) -> Option<String> {
    OVERRIDES.with_borrow(|s| {
        for scope in s.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    })
    .or_else(|| std::env::var(name).ok())
}

/// tinygrad `getenv(env, default)` when default is an int (incl. bool).
pub fn getenv_int(name: &str, default: i64) -> i64 {
    match lookup(name) {
        None => default,
        // empty string is falsy in tinygrad's getenv for ints
        Some(v) if v.is_empty() => 0,
        Some(v) => v.parse::<i64>().unwrap_or_else(|_| if v == "1" { 1 } else { 0 }),
    }
}

/// tinygrad `getenv(env, default)` when default is a str.
pub fn getenv_str(name: &str, default: &str) -> String {
    match lookup(name) {
        None => default.to_string(),
        Some(v) => v,
    }
}

/// tinygrad `getenv(env, default)` when default is a float.
pub fn getenv_float(name: &str, default: f64) -> f64 {
    match lookup(name) {
        None => default,
        Some(v) => v.parse::<f64>().unwrap_or(default),
    }
}

/// `ContextVar` for typed defaults (e.g. DEFAULT_FLOAT).
pub struct ContextVar {
    pub env: &'static str,
    pub default: &'static str,
}
impl ContextVar {
    pub const fn new(env: &'static str, default: &'static str) -> Self {
        ContextVar { env, default }
    }
    /// `.value` — returns the string value (dtype attr name for DEFAULT_*).
    pub fn value(&self) -> String {
        getenv_str(self.env, self.default)
    }
    /// Set an override at the top of the current scope stack (context-manager substitute).
    pub fn set(&self, v: &str) {
        push_context(&[(self.env, v.to_string())]);
    }
    pub fn unset(&self) {
        pop_context();
    }
}

pub static DEFAULT_FLOAT: ContextVar = ContextVar::new("DEFAULT_FLOAT", "float");
pub static DEFAULT_INT: ContextVar = ContextVar::new("DEFAULT_INT", "int");
pub static SUM_DTYPE: ContextVar = ContextVar::new("SUM_DTYPE", "float");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_scopes() {
        assert_eq!(getenv_int("ARUST_TEST_INT", 7), 7);
        push_context(&[("ARUST_TEST_INT", "3".to_string())]);
        assert_eq!(getenv_int("ARUST_TEST_INT", 7), 3);
        push_context(&[("ARUST_TEST_INT", "9".to_string())]);
        assert_eq!(getenv_int("ARUST_TEST_INT", 7), 9);
        pop_context();
        assert_eq!(getenv_int("ARUST_TEST_INT", 7), 3);
        pop_context();
        assert_eq!(getenv_int("ARUST_TEST_INT", 7), 7);
    }
}
