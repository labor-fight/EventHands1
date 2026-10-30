// Constant-albedo MANO triangle coverage.  No learned or mutable global state.
// The reference renderer is a depth-buffer renderer with one common colour for
// every triangle.  With all vertices beyond the near plane, its intensity is
// foreground iff at least one projected triangle covers the pixel.  The union
// below is therefore the same image, without redundant depth/color arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" int s43_constant_coverage(const double* uv, const std::int64_t* faces,
                                    int frames, int vertices, int triangles,
                                    int height, int width, std::uint8_t* output) {
    if (!uv || !faces || !output || frames < 1 || vertices < 3 || triangles < 1 ||
        height < 1 || width < 1) return 1;
    std::memset(output, 0, static_cast<std::size_t>(frames) * height * width);
    for (int n = 0; n < frames; ++n) {
        const double* xy = uv + static_cast<std::size_t>(n) * vertices * 2;
        std::uint8_t* mask = output + static_cast<std::size_t>(n) * height * width;
        for (int t = 0; t < triangles; ++t) {
            const std::int64_t a = faces[3*t], b = faces[3*t+1], c = faces[3*t+2];
            const double x0=xy[2*a], y0=xy[2*a+1];
            const double x1=xy[2*b], y1=xy[2*b+1];
            const double x2=xy[2*c], y2=xy[2*c+1];
            const double xmin=std::min({x0,x1,x2}), xmax=std::max({x0,x1,x2});
            const double ymin=std::min({y0,y1,y2}), ymax=std::max({y0,y1,y2});
            if (xmax < 0. || ymax < 0. || xmin > width-1 || ymin > height-1) continue;
            const int left=static_cast<int>(std::max(0.,std::ceil(xmin)));
            const int right=static_cast<int>(std::min(double(width-1),std::floor(xmax)));
            const int top=static_cast<int>(std::max(0.,std::ceil(ymin)));
            const int bottom=static_cast<int>(std::min(double(height-1),std::floor(ymax)));
            if (left > right || top > bottom) continue;
            const double den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);
            if (std::abs(den) < 1e-14) continue;
            for (int y=top; y<=bottom; ++y) for (int x=left; x<=right; ++x) {
                const double w0=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den;
                const double w1=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den;
                const double w2=1.-w0-w1;
                if (w0 >= -1e-9 && w1 >= -1e-9 && w2 >= -1e-9)
                    mask[static_cast<std::size_t>(y)*width+x]=1;
            }
        }
    }
    return 0;
}
