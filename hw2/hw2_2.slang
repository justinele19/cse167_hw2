// Eq. 4 edge function = 2 x signed area of triangle (a, b, p)
// (same helper as hw2_1.slang)
float edge(float2 a, float2 b, float2 p) {
    float2 e = b - a;
    float2 d = p - a;
    return d.x * e.y - d.y * e.x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    float2 p = float2(ipixel) + 0.5;                // pixel center
    float area = edge(sp0, sp1, sp2);               // 2 x signed area of the whole triangle
    if (area == 0) return;                          // degenerate triangle: avoid dividing by zero

    // Eq. 6: projected barycentrics b' as signed area ratios (signs cancel, so winding doesn't matter)
    // b0 = how much of vertex 0 is at p (area of the sub-triangle opposite vertex 0), etc. They sum to 1.
    float b0 = edge(sp1, sp2, p) / area;
    float b1 = edge(sp2, sp0, p) / area;
    float b2 = edge(sp0, sp1, p) / area;
    if (b0 < 0 || b1 < 0 || b2 < 0) return;         // outside the triangle

    // Eq. 15: weight by inverse depth, then normalize
    // (screen-space barycentrics are skewed by perspective; 1/depth is what interpolates linearly on screen)
    float w0 = b0 * inv_depth0;
    float w1 = b1 * inv_depth1;
    float w2 = b2 * inv_depth2;
    float sum = w0 + w1 + w2;
    float depth = 1.0 / sum;                        // Eq. 16 simplified: this is -z at the pixel

    // z-buffer test: depth_image starts at infinity, so the first triangle always passes
    if (depth < depth_image[ipixel]) {              // closer than what's stored? then this triangle wins
        depth_image[ipixel] = depth;
        color_image[ipixel] = float4((w0 * c0 + w1 * c1 + w2 * c2) / sum, 1.0);   // Eq. 17, perspective-correct
    }
}
