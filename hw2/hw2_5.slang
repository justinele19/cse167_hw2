// =============================================================================
// HW 2.5: Parallel software rasterizer (one GPU thread per triangle)
//
// Before (2.2-2.4): Python launched ONE kernel PER TRIANGLE, with a barrier after
// each one. With 871k dragon triangles that is 871k launches in a row: slow.
//
// Now: Python launches each kernel ONCE, and every GPU thread handles one item.
// Per frame, hw2.py runs these kernels in order:
//   1. transform_vertices  one thread per VERTEX:   object space -> screen space
//   2. clear_buffers       one thread per PIXEL:    reset depth + triangle ID
//   3. rasterize_depth     one thread per TRIANGLE: find the closest depth per pixel
//   4. rasterize_id        one thread per TRIANGLE: find WHICH triangle has that depth
//   5. shade               one thread per PIXEL:    color the pixel from that triangle
// (with a global_barrier between steps so each one sees the previous one's results)
//
// Why two triangle passes (3 and 4) instead of one?
//   Many threads can hit the same pixel at the same time, so writes must be
//   ATOMIC. GPUs only reliably support 32-bit integer atomics, so we can't
//   atomically update "depth AND color" together. Instead:
//   pass 3 atomically keeps the smallest depth, pass 4 atomically records which
//   triangle produced it (the "visibility buffer"), and pass 5 colors the pixel
//   afterwards ("deferred shading").
//
// Buffers are flat 1D arrays of size width*height; pixel (x, y) is index y*width + x.
// =============================================================================


// Eq. 4 edge function = 2 x signed area of triangle (a, b, p)
// The SIGN says which side of the line a->b the point p is on:
//   > 0 on one side, < 0 on the other, = 0 exactly on the line.
// This is a 2D cross product of (b - a) and (p - a).
float edge(float2 a, float2 b, float2 p) {
    float2 e = b - a;   // direction of the edge
    float2 d = p - a;   // from the edge start to the point
    return d.x * e.y - d.y * e.x;
}

// Shared by all three kernels so the depth bits match exactly between passes
// and degenerate triangles are rejected the same way everywhere.
// (rasterize_id compares depths with ==, so pass 3 and pass 4 must compute
//  the exact same float. Using one function guarantees identical math.)
// v.xy = screen position, v.z = inverse depth.
// Returns false if the triangle is degenerate or the pixel center is outside;
// otherwise w = normalized perspective-correct weights and depth = -z at the pixel.
bool cover(float3 v0, float3 v1, float3 v2, int2 pixel, out float3 w, out float depth) {
    // "out" parameters must always be assigned, even when we return false
    w = float3(0, 0, 0);
    depth = 0;

    float2 p = float2(pixel) + 0.5;                 // test the pixel CENTER, not its corner
    float area = edge(v0.xy, v1.xy, v2.xy);         // 2x signed area of the whole triangle
    if (area == 0) return false;                    // degenerate triangle (zero area: would divide by 0)

    // Eq. 6: projected barycentrics
    // Each b_i = area of the sub-triangle opposite vertex i / total area.
    // Dividing by the SIGNED area cancels the sign, so this works for
    // clockwise and counter-clockwise triangles. b0 + b1 + b2 = 1.
    float b0 = edge(v1.xy, v2.xy, p) / area;
    float b1 = edge(v2.xy, v0.xy, p) / area;
    float b2 = edge(v0.xy, v1.xy, p) / area;
    if (b0 < 0 || b1 < 0 || b2 < 0) return false;   // outside the triangle (a negative weight = wrong side of an edge)

    // Eq. 15: weight by inverse depth, then normalize
    // Screen-space barycentrics are NOT correct in 3D after perspective
    // (far things get squished). Weighting each one by 1/depth fixes that:
    // this is "perspective-correct interpolation".
    float3 wi = float3(b0 * v0.z, b1 * v1.z, b2 * v2.z);
    float sum = wi.x + wi.y + wi.z;                 // = interpolated 1/depth at this pixel
    depth = 1.0 / sum;                              // flip back to get the actual depth (-z), always positive
    w = wi / sum;                                   // normalized so the weights sum to 1
    return true;
}

// Screen space AABB: floor the min, ceil the max, clamped to the image
// (AABB = axis-aligned bounding box: the smallest rectangle around the triangle.
//  Only pixels inside it can possibly be covered, so we only loop over those.)
void screen_aabb(float3 v0, float3 v1, float3 v2, uint width, uint height,
                 out int2 aabb_min, out int2 aabb_max) {
    float2 lo = floor(min(min(v0.xy, v1.xy), v2.xy));         // top-left corner (component-wise min)
    float2 hi = ceil(max(max(v0.xy, v1.xy), v2.xy));          // bottom-right corner (component-wise max)
    aabb_min = max(int2(lo), int2(0, 0));                     // never left of / above the image
    aabb_max = min(int2(hi), int2(width, height) - 1);        // never past the last pixel
    // If the triangle is fully off-screen, min ends up > max and the loops simply don't run.
}

// [Starter code] Step 2: reset every pixel before rasterizing.
// 0xffffffff is the LARGEST uint, so any real depth / triangle ID will be smaller
// and win the InterlockedMin. It also means "no triangle" in the ID buffer.
[shader("compute")]
[numthreads(64, 1, 1)]                  // threads run in groups of 64; tid = this thread's global index
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)              // extra threads in the last group of 64 do nothing
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

// [Starter code] Step 1: the vertex projection, moved from Python onto the GPU.
// One thread per vertex. Python launches this once PER MESH because each mesh
// has its own MVP (model matrix); vertex_offset says where that mesh's vertices
// start in the merged vertex array.
// With our projection matrix, p.w = -z (the depth), so p.xyz / p.w gives:
//   x, y = pixel coordinates,  z = 1 / depth (the "inverse depth" used by cover()).
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

    uint i = vertex_offset + tid;                       // index into the merged array
    float4 p = mul(MVP, float4(vertices[i], 1.0));      // homogeneous point (w = 1)
    projected_vertices[i] = p.xyz / p.w;                // homogeneous divide
}

// Step 3: DEPTH PASS. One thread per triangle.
// Each thread loops over the pixels its triangle covers and atomically keeps the
// SMALLEST depth (closest surface) in depth_buffer.
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,     // = this thread's triangle index
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,                    // (unused here; kept from the starter signature)
        uniform float a,                    // (unused here; kept from the starter signature)
        StructuredBuffer<uint3> faces,      // each face = 3 vertex indices
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO
    if (tid >= num_triangles)               // extra threads in the last group of 64
        return;

    // 1. Fetch the three screen-space vertices of this triangle
    uint3 f = faces[tid];
    float3 v0 = projected_vertices[f.x];
    float3 v1 = projected_vertices[f.y];
    float3 v2 = projected_vertices[f.z];

    // 2. Compute the screen-space bounding box
    int2 aabb_min, aabb_max;
    screen_aabb(v0, v1, v2, width, height, aabb_min, aabb_max);

    // 3. Iterate over the pixels inside the bounding box
    for (int y = aabb_min.y; y <= aabb_max.y; y++) {
        for (int x = aabb_min.x; x <= aabb_max.x; x++) {
            float3 w;
            float depth;
            // 4. Is this pixel center inside the triangle?
            if (cover(v0, v1, v2, int2(x, y), w, depth)) {
                // 5. Keep the closest depth.
                // positive floats order the same as their bits, so a uint min works
                // asuint() REINTERPRETS the float's 32 bits as a uint (no rounding/conversion).
                // InterlockedMin does "buffer = min(buffer, value)" as ONE indivisible step,
                // so two threads writing the same pixel at once can't overwrite each other.
                InterlockedMin(depth_buffer[y * width + x], asuint(depth));
            }
        }
    }
}

// Step 4: ID PASS (builds the "visibility buffer"). One thread per triangle.
// Same loop as the depth pass, but now depth_buffer is final. A triangle whose
// depth at a pixel EQUALS the stored minimum is the visible one there, so it
// writes its triangle ID into triangle_id_buffer.
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,     // = this thread's triangle index (= the face ID we store)
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
            // Covered AND exactly the closest depth (compared as bits, same as pass 3 stored)?
            if (cover(v0, v1, v2, int2(x, y), w, depth) &&
                asuint(depth) == depth_buffer[idx]) {
                // atomic min breaks ties between triangles with identical depth
                // (two triangles at the exact same depth -> the smaller ID wins, deterministically)
                InterlockedMin(triangle_id_buffer[idx], tid);
            }
        }
    }
}

// Step 5: SHADING PASS ("deferred shading"). One thread per PIXEL.
// We already know which triangle is visible at each pixel, so each pixel is
// colored exactly once, using only the winning triangle.
[shader("compute")]
[numthreads(8, 8, 1)]                       // 2D groups of 8x8 pixels; tid = (x, y)
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
    if (tid.x >= width || tid.y >= height) {    // threads past the image edge do nothing
        return;
    }

    // HW 2.5 TODO
    uint face_id = triangle_id_buffer[tid.y * width + tid.x];   // which triangle won this pixel
    if (face_id == 0xffffffff)
        return;                                     // no triangle here: keep the background

    uint3 f = faces[face_id];
    float3 w;
    float depth;
    // Recompute the perspective-correct weights for this pixel (we only stored the ID).
    // cover() returns true here, since this triangle passed the same test in pass 4.
    if (cover(projected_vertices[f.x], projected_vertices[f.y], projected_vertices[f.z],
              int2(tid), w, depth)) {
        // Eq. 17, perspective-correct color
        // Blend the three vertex colors with the weights.
        float3 color = w.x * vertex_colors[f.x] + w.y * vertex_colors[f.y] + w.z * vertex_colors[f.z];
        image[tid] = float4(color, 1.0);            // alpha = 1 (opaque)
    }
}
