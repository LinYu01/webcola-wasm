## fix build

```
rustup default nightly
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.73
cargo install wasm-opt
```

## patch by hand 

```js
export function create_derivative_computer_ctx_2d(node_count, D, G) {
    var ptr0 = passArrayF32ToWasm0(D, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    var ptr1 = passArrayF32ToWasm0(G, wasm.__wbindgen_malloc);
    var len1 = WASM_VECTOR_LEN;
    var ret = wasm.create_derivative_computer_ctx_2d(node_count, ptr0, len0, ptr1, len1);
    return [ret, ptr0, ptr1];
}

/**
* @param {number} node_count
* @param {Float32Array} D
* @param {Float32Array} G
* @returns {number}
*/
export function create_derivative_computer_ctx_3d(node_count, D, G) {
    var ptr0 = passArrayF32ToWasm0(D, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    var ptr1 = passArrayF32ToWasm0(G, wasm.__wbindgen_malloc);
    var len1 = WASM_VECTOR_LEN;
    var ret = wasm.create_derivative_computer_ctx_3d(node_count, ptr0, len0, ptr1, len1);
    return [ret, ptr0, ptr1];
}

/**
* @param {number} ctx
* @param {Float32Array} new_G
*/
export function set_G_2d(ctx, new_G) {
    var ptr0 = passArrayF32ToWasm0(new_G, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    wasm.set_G_2d(ctx, ptr0, len0);
    return ptr0;
}

/**
* @param {number} ctx
* @param {Float32Array} new_G
*/
export function set_G_3d(ctx, new_G) {
    var ptr0 = passArrayF32ToWasm0(new_G, wasm.__wbindgen_malloc);
    var len0 = WASM_VECTOR_LEN;
    wasm.set_G_3d(ctx, ptr0, len0);
    return ptr0;
}
```