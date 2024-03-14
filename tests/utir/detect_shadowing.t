// RUN: @tara @file
// CHECK:[0m[1m[38;5;9merror[0m[1m: Variable declaration shadows previous declaration[0m
// CHECK:   tests/utir/detect_shadowing.t:14:1
// CHECK:   [0m[34m│[0m  
// CHECK:[0m[34m11[0m [0m[34m│[0m [0m[34m╭[0m const S = struct {
// CHECK:[0m[34m12[0m [0m[34m│[0m [0m[34m│[0m     const S = struct {};
// CHECK:[0m[34m13[0m [0m[34m│[0m [0m[34m│[0m };
// CHECK:   [0m[34m│[0m [0m[34m╰[0m[0m[34m──' original declared here[0m
// CHECK:[0m[34m14[0m [0m[34m│[0m   [0m[31mconst S = struct {};[0m
// CHECK:   [0m[34m│[0m   [0m[31m^^^^^^^^^^^^^^^^^^^^[0m [0m[31mshadow declared here[0m
const S = struct {
    const S = struct {};
};
const S = struct {};
