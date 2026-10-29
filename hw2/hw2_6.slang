// =============================================================================
// HW 2.6: Hardware rasterization (the GPU's built-in graphics pipeline)
//
// Instead of writing the rasterizer ourselves (2.1-2.5), we only write the two
// programmable stages; the GPU does everything in between:
//   1. vertex_main    (ours)  runs once per vertex: object space -> clip space
//   2. clip + raster  (GPU)   clips triangles to the view, does the divide by w,
//                             maps to pixels, finds covered pixels, interpolates
//                             VertexOutput across each triangle
//   3. fragment_main  (ours)  runs once per covered pixel: returns its color
//   4. depth test     (GPU)   keeps the closest fragment (set up in hw2.py)
// =============================================================================

// Inputs bound from Python (hw2_6 in hw2.py) via the ShaderCursor
StructuredBuffer<float3> vertices;      // this mesh's vertex positions (object space)
StructuredBuffer<float3> vertex_colors; // one RGB color per vertex
uniform float4x4 MVP;                   // clip_projection @ view @ model (same for every vertex)

// What the vertex shader hands to the GPU for each vertex.
// The ": NAME" parts are "semantics" that tell the GPU what each field is.
struct VertexOutput {
    float4 position : SV_Position;  // REQUIRED: clip-space position (x, y, z, w)
    float3 color : COLOR0;          // extra data; the GPU interpolates it per pixel
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.6: your code here
    // vid comes from the index buffer, so it indexes the vertex arrays directly
    // (draw_indexed reads the triangle indices and calls us with each vertex index)
    output.position = mul(MVP, float4(vertices[vid], 1.0));   // clip space; the GPU does the divide by w
    // Not dividing by w here is required: the GPU needs w to clip triangles that
    // cross the near plane and to do perspective-correct interpolation.
    output.color = vertex_colors[vid];                         // interpolated for the fragment shader
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {   // SV_Target = write to the color image
    // HW 2.6: your code here
    // input.color has already been blended from the triangle's 3 vertex colors
    // (the same barycentric interpolation we did by hand in 2.2-2.5).
    return float4(input.color, 1.0);   // already perspective-correct interpolated by the GPU
}
