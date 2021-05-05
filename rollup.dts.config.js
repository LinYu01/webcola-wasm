import dts from "rollup-plugin-dts";
import wasm from "@rollup/plugin-wasm";

export default [
  {
    input: "./tmp/index.d.ts",
    output: [{
      file: "dist/cola.d.ts",
      format: "esm",
      banner: 'export as namespace cola;'
    }],
    plugins: [dts(), wasm()],
  },
];
