## fix build

```
rustup default nightly
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.73
# we use this version now
cargo install -f wasm-bindgen-cli --version 0.2.87
cargo install wasm-opt
```

## known issues

build will have this ts error, the final result is fine.

```
Error: Could not resolve './wasm/build/simd/derivative_computer' from tmp/wasmEngine.d.ts
```

## ref

https://github.com/rustwasm/wasm-bindgen/issues/2957
https://github.com/rustwasm/wasm-bindgen/issues?q=Box%3A%3Ainto_raw
