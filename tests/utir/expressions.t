// RUN: @tara @file --dump-utir
// CHECK: test

const A = u1;
const B = a & b;
const C = -a;
const D = a.b();
const E = a.b(c, d);
const F = (a | b);
const G = &u1;
const H = &var u1;
const I = undefined;
const J = true;
const K = false;
const L = a.*;
const M = return a;
const N = 0;
const O = 0b0;
const P = 0xF;
const Q = 3b111;
const R = 32xFFFFFFFF;
const S = if (a) e else b;
