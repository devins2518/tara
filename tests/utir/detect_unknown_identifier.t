// RUN: @tara @file --dump-utir
// CHECK:[0m[1m[38;5;9merror[0m[1m: Use of unknown identifier[0m
// CHECK:  [0m[34m┌─[0m tests/utir/detect_unknown_identifier.t:7:11
// CHECK:  [0m[34m│[0m
// CHECK:[0m[34m7[0m [0m[34m│[0m const A = [0m[31ma[0m;
// CHECK:  [0m[34m│[0m           [0m[31m^[0m [0m[31munknown identifier used here[0m
const A = a;
