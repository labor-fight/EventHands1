// Fixed-kernel, lazy sensor-memory update. No MANO or learned state is used.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

template <typename F>
int update(F* A, F* Z, int64_t* ta, int64_t* tz,
           const int32_t* src, const int32_t* incoming, int C, int degree,
           int R, int L, const double* tau, const double* gamma,
           const int64_t* group_t, const int64_t* offsets, int groups,
           const int32_t* cells, const int64_t* counts,
           int64_t* totals, int64_t* first, int64_t* last) {
    // Counts are grouped by timestamp then cell, with two polarity columns.
    std::vector<double> da(R), dz(L);
    // Integer-ms streams overwhelmingly revisit a small set of exact ages.
    // This is an exact exp lookup, not a truncation of the continuous tail;
    // arbitrary microsecond or older ages still use the analytic expression.
    constexpr int cache_ms=256;
    std::vector<double> ca((cache_ms+1)*R), cz((cache_ms+1)*L);
    for (int k=0;k<=cache_ms;++k) {
        for (int r=0;r<R;++r) ca[k*R+r]=std::exp(-double(k*1000)/tau[r]);
        for (int l=0;l<L;++l) cz[k*L+l]=std::exp(-double(k*1000)/gamma[l]);
    }
    auto decay_a = [&](int a, int64_t t) {
        const int64_t dt = t - ta[a];
        if (dt != 0) {
            const bool cached=dt>=0 && dt<=cache_ms*1000 && dt%1000==0;
            for (int r=0;r<R;++r)
                da[r] = cached ? ca[(dt/1000)*R+r] : std::exp(-double(dt)/tau[r]);
            for (int p=0;p<2;++p) for (int r=0;r<R;++r)
                A[(a*2+p)*R+r] *= F(da[r]);
            ta[a] = t;
        }
    };
    for (int g=0;g<groups;++g) {
        const int64_t t=group_t[g];
        for (int64_t j=offsets[g];j<offsets[g+1];++j) {
            const int b=cells[j];
            for (int k=0;k<degree;++k) {
                const int e=incoming[b*degree+k];
                if (e<0) continue;
                const int a=src[e];
                decay_a(a,t);  // decay only: all current-group additions wait
                const int64_t dt=t-tz[e];
                const bool cached=dt>=0 && dt<=cache_ms*1000 && dt%1000==0;
                for (int l=0;l<L;++l)
                    dz[l]=cached ? cz[(dt/1000)*L+l] : std::exp(-double(dt)/gamma[l]);
                for (int p=0;p<2;++p) for (int q=0;q<2;++q)
                    for (int r=0;r<R;++r) for (int l=0;l<L;++l) {
                        const int64_t z=((((int64_t(e)*2+p)*2+q)*R+r)*L+l);
                        const double v=double(Z[z])*dz[l]
                            +double(A[(a*2+p)*R+r])*double(counts[j*2+q]);
                        if (!std::isfinite(v) || v>std::numeric_limits<F>::max()) return 1;
                        Z[z]=F(v);
                    }
                tz[e]=t;
            }
        }
        for (int64_t j=offsets[g];j<offsets[g+1];++j) {
            const int a=cells[j];
            decay_a(a,t);
            for (int p=0;p<2;++p) {
                const int64_t n=counts[j*2+p];
                if (n==0) continue;
                const int64_t cp=a*2+p;
                if (totals[cp]>std::numeric_limits<int64_t>::max()-n) return 2;
                totals[cp]+=n;
                if (first[cp]<0) first[cp]=t;
                last[cp]=t;
                for (int r=0;r<R;++r) {
                    const double v=double(A[cp*R+r])+double(n);
                    if (!std::isfinite(v) || v>std::numeric_limits<F>::max()) return 1;
                    A[cp*R+r]=F(v);
                }
            }
        }
    }
    return 0;
}

extern "C" {
#define ARGS(F) F* A,F* Z,int64_t* ta,int64_t* tz,const int32_t* src,const int32_t* incoming,int C,int degree,int R,int L,const double* tau,const double* gamma,const int64_t* group_t,const int64_t* offsets,int groups,const int32_t* cells,const int64_t* counts,int64_t* totals,int64_t* first,int64_t* last
#define PASS A,Z,ta,tz,src,incoming,C,degree,R,L,tau,gamma,group_t,offsets,groups,cells,counts,totals,first,last
int rel_update_f32(ARGS(float)) { return update(PASS); }
int rel_update_f64(ARGS(double)) { return update(PASS); }
}
