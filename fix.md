## fix build

```
rustup default nightly
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.73
cargo install wasm-opt
```
