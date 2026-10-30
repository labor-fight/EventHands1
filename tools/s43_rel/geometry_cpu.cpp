// Exact interval subtraction for the silhouette of a projected triangle union.
// No image, event, radiance, GT visibility, or raster finite differences enter.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

extern "C" int s43_rel_visible_segments(
    const double* xy, const int64_t* faces, int nf,
    const int64_t* edges, int ne, double width, double height,
    double epsilon, double* output, int capacity) {
  std::vector<double> answer;
  struct Triangle {double xmin,xmax,ymin,ymax,area;};
  std::vector<Triangle> boxes;
  boxes.reserve(nf);
  for(int f=0;f<nf;++f) {
    const int64_t* ff=faces+3*f;
    double ax=xy[2*ff[0]],ay=xy[2*ff[0]+1];
    double bx=xy[2*ff[1]],by=xy[2*ff[1]+1];
    double cx=xy[2*ff[2]],cy=xy[2*ff[2]+1];
    boxes.push_back({std::min({ax,bx,cx}),std::max({ax,bx,cx}),
                     std::min({ay,by,cy}),std::max({ay,by,cy}),
                     (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)});
  }
  for (int e = 0; e < ne; ++e) {
    int a = edges[2*e], b = edges[2*e+1];
    double x = xy[2*a], y = xy[2*a+1];
    double dx = xy[2*b]-x, dy = xy[2*b+1]-y;
    double lo = 0., hi = 1.;
    // A visible MANO edge is clipped to the sensor, but image borders never
    // become artificial moving silhouette generators.
    const double aa[4] = {x, width-1.-x, y, height-1.-y};
    const double bb[4] = {dx, -dx, dy, -dy};
    bool valid = true;
    for (int q=0;q<4;++q) {
      if (std::abs(bb[q]) < 1e-14) { if (aa[q]<0.) valid=false; }
      else if (bb[q]>0.) lo=std::max(lo,-aa[q]/bb[q]);
      else hi=std::min(hi,-aa[q]/bb[q]);
    }
    if (!valid || hi<=lo) continue;
    double sxmin=std::min(x+lo*dx,x+hi*dx),sxmax=std::max(x+lo*dx,x+hi*dx);
    double symin=std::min(y+lo*dy,y+hi*dy),symax=std::max(y+lo*dy,y+hi*dy);
    std::vector<std::pair<double,double>> covered;
    for (int f=0;f<nf;++f) {
      const Triangle& box=boxes[f];
      if(box.xmin>sxmax || box.xmax<sxmin || box.ymin>symax || box.ymax<symin) continue;
      const int64_t* ff = faces+3*f;
      double area=box.area;
      if (std::abs(area)<epsilon) continue;
      double sign=area>0.?1.:-1., tl=lo, th=hi;
      bool interior=true;
      for (int q=0;q<3;++q) {
        int u=ff[q], v=ff[(q+1)%3];
        double ux=xy[2*u], uy=xy[2*u+1];
        double ex=xy[2*v]-ux, ey=xy[2*v+1]-uy;
        double A=sign*(ex*(y-uy)-ey*(x-ux));
        double B=sign*(ex*dy-ey*dx);
        // Strict interior excludes the generating edge's adjacent triangles.
        // Offset epsilon is a numerical tolerance, not a visibility dilation.
        double tol=epsilon*std::max(1.,std::hypot(ex,ey));
        if (std::abs(B)<1e-14) { if(A<=tol) {interior=false;break;} }
        else if(B>0.) tl=std::max(tl,(tol-A)/B);
        else th=std::min(th,(tol-A)/B);
      }
      if(interior && th>tl) covered.emplace_back(tl,th);
    }
    std::sort(covered.begin(),covered.end());
    double start=lo;
    for (auto interval:covered) {
      if(interval.first>start+1e-10) {
        answer.insert(answer.end(),{double(e),start,interval.first});
      }
      start=std::max(start,interval.second);
      if(start>=hi) break;
    }
    if(start<hi-1e-10) answer.insert(answer.end(),{double(e),start,hi});
  }
  int n=int(answer.size()/3);
  if(n>capacity) return -n;
  std::copy(answer.begin(),answer.end(),output);
  return n;
}
