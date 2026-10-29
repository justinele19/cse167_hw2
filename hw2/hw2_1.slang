// Eq. 4: E(p) = (p - a) . n, where n = (e.y, -e.x) and e = b - a
// The sign says which side of the line a->b the point p is on (0 = exactly on the line)
float edge(float2 a, float2 b, float2 p) {
    float2 e = b - a;
    float2 d = p - a;
    return d.x * e.y - d.y * e.x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    // each thread handles one pixel in the AABB (ipixel is computed above)
    float2 p = float2(ipixel) + 0.5;          // test the pixel CENTER
    // which side of each of the 3 edges is p on?
    float e01 = edge(sp0, sp1, p);
    float e12 = edge(sp1, sp2, p);
    float e20 = edge(sp2, sp0, p);

    // inside = all three the same sign; checking both signs covers clockwise AND counter-clockwise triangles
    if ((e01 >= 0 && e12 >= 0 && e20 >= 0) || (e01 <= 0 && e12 <= 0 && e20 <= 0))
        image[ipixel] = float4(color, 1.0);
}
