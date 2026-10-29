// Eq. 4 edge function = 2 x signed area of triangle (a, b, p)
float edge(float2 a, float2 b, float2 p) {
    float2 e = b - a;
    float2 d = p - a;
    return d.x * e.y - d.y * e.x;
}

// Shared by all three kernels so the depth bits match exactly between passes
// and degenerate triangles are rejected the same way everywhere.
// v.xy = screen position, v.z = inverse depth.
// Returns false if the triangle is degenerate or the pixel center is outside;
// otherwise w = normalized perspective-correct weights and depth = -z at the pixel.
bool cover(float3 v0, float3 v1, float3 v2, int2 pixel, out float3 w, out float depth) {
    w = float3(0, 0, 0);
    depth = 0;
    float2 p = float2(pixel) + 0.5;
    float area = edge(v0.xy, v1.xy, v2.xy);
    if (area == 0) return false;                    // degenerate triangle

    // Eq. 6: projected barycentrics
    float b0 = edge(v1.xy, v2.xy, p) / area;
    float b1 = edge(v2.xy, v0.xy, p) / area;
    float b2 = edge(v0.xy, v1.xy, p) / area;
    if (b0 < 0 || b1 < 0 || b2 < 0) return false;   // outside the triangle

    // Eq. 15: weight by inverse depth, then normalize
    float3 wi = float3(b0 * v0.z, b1 * v1.z, b2 * v2.z);
    float sum = wi.x + wi.y + wi.z;
    depth = 1.0 / sum;
    w = wi / sum;
    return true;
}

// Screen space AABB: floor the min, ceil the max, clamped to the image
void screen_aabb(float3 v0, float3 v1, float3 v2, uint width, uint height,
                 out int2 aabb_min, out int2 aabb_max) {
    float2 lo = floor(min(min(v0.xy, v1.xy), v2.xy));
    float2 hi = ceil(max(max(v0.xy, v1.xy), v2.xy));
    aabb_min = max(int2(lo), int2(0, 0));
    aabb_max = min(int2(hi), int2(width, height) - 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO
    if (tid >= num_triangles)
        return;

    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    int2 aabb_min, aabb_max;
    screen_aabb(v0, v1, v2, width, height, aabb_min, aabb_max);

    for (int y = aabb_min.y; y <= aabb_max.y; y++) {
        for (int x = aabb_min.x; x <= aabb_max.x; x++) {
            float3 w;
            float depth;
            if (cover(v0, v1, v2, int2(x, y), w, depth)) {
                // positive floats order the same as their bits, so a uint min works
                InterlockedMin(depth_buffer[y * width + x], asuint(depth));
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO
    if (tid >= num_triangles)
        return;

    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    int2 aabb_min, aabb_max;
    screen_aabb(v0, v1, v2, width, height, aabb_min, aabb_max);

    for (int y = aabb_min.y; y <= aabb_max.y; y++) {
        for (int x = aabb_min.x; x <= aabb_max.x; x++) {
            float3 w;
            float depth;
            uint idx = y * width + x;
            if (cover(v0, v1, v2, int2(x, y), w, depth) &&
                asuint(depth) == depth_buffer[idx]) {
                // atomic min breaks ties between triangles with identical depth
                InterlockedMin(triangle_id_buffer[idx], tid);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // HW 2.5 TODO
    uint face_id = triangle_id_buffer[tid.y * width + tid.x];
    if (face_id == 0xffffffff)
        return;                                     // no triangle here: keep the background

    uint3 f = faces[face_id];
    float3 w;
    float depth;
    if (cover(projected_vertices[f.x], projected_vertices[f.y], projected_vertices[f.z],
              int2(tid), w, depth)) {
        // Eq. 17, perspective-correct color
        float3 color = w.x * vertex_colors[f.x] + w.y * vertex_colors[f.y] + w.z * vertex_colors[f.z];
        image[tid] = float4(color, 1.0);
    }
}
