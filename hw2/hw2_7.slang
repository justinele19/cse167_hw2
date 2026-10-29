// =============================================================================
// HW 2.7: Phong lighting + textures (same GPU pipeline as 2.6)
//
// Vertex shader: positions -> clip space, and also hands the fragment shader
//                world-space position, world-space normal, and texture coords.
// Fragment shader: computes the color of each pixel with Phong lighting:
//                  C = ambient + diffuse + specular
// Lighting is done in WORLD space because the light direction and the camera
// position are given in world space.
// =============================================================================

// Per-mesh data, bound from Python (hw2_7 in hw2.py)
StructuredBuffer<float3> vertices;          // positions (object space)
StructuredBuffer<float3> vertex_normals;    // one normal per vertex (object space)
StructuredBuffer<float2> vertex_uvs;        // texture coordinates (u, v) in [0,1]^2
uniform float4x4 MVP;                       // object -> clip space
uniform float4x4 model_matrix;              // object -> world space
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
// Scene / material data
uniform float3 light_dir;                   // l: points FROM the scene TOWARD the light (directional, like the sun)
uniform float3 light_color;                 // E: light color/brightness
uniform float3 camera_position;             // camera position in world space
uniform float3 Ks;                          // specular reflectance (shine color)
uniform float exponent;                     // Phong exponent e: bigger = smaller, sharper highlight
Texture2D<float4> Kd_texture;               // diffuse color (a 1x1 texture if Kd is a constant)
SamplerState Kd_sampler;                    // how to read the texture (filtering)

// Everything here gets interpolated across the triangle by the GPU.
// TEXCOORDn are just generic "slots" for our own data.
struct VertexOutput {
    float4 position : SV_Position;          // clip space (for the GPU)
    float3 world_position : TEXCOORD0;      // for the view vector v
    float3 world_normal : TEXCOORD1;        // for n (Phong's trick: interpolated vertex normals = smooth shading)
    float2 uv : TEXCOORD2;                  // for the texture lookup
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.7 TODO: implement
    float4 p = float4(vertices[vid], 1.0);                          // homogeneous point (w = 1)
    output.position = mul(MVP, p);                                  // clip space; the GPU does the divide by w
    output.world_position = mul(model_matrix, p).xyz;               // model matrix is affine, so w stays 1
    output.world_normal = mul(normal_matrix, vertex_normals[vid]);  // normals need the inverse transpose
    // (Using the model matrix on normals would tilt them wrong under non-uniform
    //  scaling; N = (M3x3^-1)^T keeps them perpendicular to the surface.)
    output.uv = vertex_uvs[vid];                                    // passed through unchanged
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.7 TODO: implement
    // --- The unit vectors Phong needs ---
    float3 n = normalize(input.world_normal);                       // interpolation doesn't preserve length
    float3 l = normalize(light_dir);                                // points from scene to light
    float3 v = normalize(camera_position - input.world_position);   // points from surface to camera
    // Diffuse color from the texture at this pixel's (u, v); sRGB -> linear happens automatically
    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    // n.l = cosine of the angle between the surface normal and the light:
    // 1 = facing the light head-on, 0 = edge-on, < 0 = facing away.
    float n_dot_l = dot(n, l);
    float3 Ca = 0.2 * Kd;                                           // ambient: Ca = 0.2 Kd (so unlit areas aren't black)
    float3 Cd = Kd * light_color * max(0.0, n_dot_l);               // diffuse: Cd = Kd * E * max(0, n.l)
    float3 Cs = float3(0.0);                                        // specular, only when lit from the front
    if (n_dot_l > 0) {
        float3 r = reflect(-l, n);                                  // mirror reflection of the light direction
        // r = 2(n.l)n - l. If r points at the camera (r.v near 1) we see a highlight;
        // raising it to the exponent makes the highlight tighter.
        // specular: Cs = Ks * E * max(0, r.v)^e
        Cs = Ks * light_color * pow(max(0.0, dot(r, v)), exponent);
    }
    return float4(Ca + Cd + Cs, 1.0);                               // C = Ca + Cd + Cs, alpha = 1
}
