StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_colors;
uniform float4x4 MVP;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.6: your code here
    // vid comes from the index buffer, so it indexes the vertex arrays directly
    output.position = mul(MVP, float4(vertices[vid], 1.0));   // clip space; the GPU does the divide by w
    output.color = vertex_colors[vid];                         // interpolated for the fragment shader
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.6: your code here
    return float4(input.color, 1.0);   // already perspective-correct interpolated by the GPU
}
