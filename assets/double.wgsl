@group(0) @binding(0)
var<uniform> input: u32;

@group(0) @binding(1)
var<storage, read_write> output: u32;

@compute @workgroup_size(1,1,1)
fn double() {
    output = input * 2u;
}