// LD_PRELOAD heap tracker: live bytes per call stack for allocations >= HT_MIN (default 64 KiB).
// On every new high of tracked live bytes (in HT_STEP MiB steps) it rewrites HT_OUT with the top stacks
// as "module+offset" frames, for offline addr2line.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define DEPTH 24
#define NSTK 65536
#define NPTR (1 << 22)
typedef struct { uint64_t h; int n; void* f[DEPTH]; int64_t live, peak_snap; } stk_t;
typedef struct { void* p; size_t sz; int s; } ent_t;
static stk_t stks[NSTK];
static ent_t ptrs[NPTR];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static __thread int guard;
static size_t min_sz = 64 << 10;
static int64_t live, hi, step = 256ll << 20, next_dump;
static const char* out = "/tmp/heaptrace.txt";
static void* (*r_malloc)(size_t); static void (*r_free)(void*); static void* (*r_calloc)(size_t, size_t);
static void* (*r_realloc)(void*, size_t); static void* (*r_mmap)(void*, size_t, int, int, int, off_t); static int (*r_munmap)(void*, size_t);
static char boot[1 << 16]; static size_t bootn;

static void init(void) {
    if (r_malloc) return;
    r_malloc = dlsym(RTLD_NEXT, "malloc"); r_free = dlsym(RTLD_NEXT, "free"); r_calloc = dlsym(RTLD_NEXT, "calloc");
    r_realloc = dlsym(RTLD_NEXT, "realloc"); r_mmap = dlsym(RTLD_NEXT, "mmap"); r_munmap = dlsym(RTLD_NEXT, "munmap");
    char* e;
    if ((e = getenv("HT_MIN"))) min_sz = strtoull(e, 0, 10);
    if ((e = getenv("HT_STEP"))) step = strtoll(e, 0, 10) << 20;
    if ((e = getenv("HT_OUT"))) out = e;
    next_dump = step;
}
static void dump(void) {
    FILE* f = fopen(out, "w"); if (!f) return;
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    fprintf(f, "TIME_MS %lld LIVE_MIB %.1f\n", (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000, live / 1048576.0);
    // top 25 stacks by live bytes (selection, small N)
    static int used[NSTK]; memset(used, 0, sizeof(used));
    for (int k = 0; k < 25; k++) {
        int best = -1;
        for (int i = 0; i < NSTK; i++) if (stks[i].n && !used[i] && (best < 0 || stks[i].live > stks[best].live)) best = i;
        if (best < 0 || stks[best].live <= 0) break;
        used[best] = 1;
        fprintf(f, "STACK %.1f MiB\n", stks[best].live / 1048576.0);
        for (int j = 2; j < stks[best].n; j++) {
            Dl_info di;
            if (dladdr(stks[best].f[j], &di) && di.dli_fname)
                fprintf(f, "  %s 0x%lx\n", di.dli_fname, (unsigned long)((char*)stks[best].f[j] - (char*)di.dli_fbase - 1));
            else fprintf(f, "  ? %p\n", stks[best].f[j]);
        }
    }
    fclose(f);
}
static void track(void* p, size_t sz) {
    if (!p || sz < min_sz || guard) return;
    guard = 1;
    void* f[DEPTH]; int n = backtrace(f, DEPTH);
    uint64_t h = 1469598103934665603ull; for (int i = 2; i < n; i++) h = (h ^ (uint64_t)f[i]) * 1099511628211ull;
    pthread_mutex_lock(&mu);
    int s = h % NSTK; while (stks[s].n && stks[s].h != h) s = (s + 1) % NSTK;
    if (!stks[s].n) { stks[s].h = h; stks[s].n = n; memcpy(stks[s].f, f, n * sizeof(void*)); }
    stks[s].live += sz; live += sz;
    uint64_t k = ((uint64_t)p >> 4) % NPTR; while (ptrs[k].p) k = (k + 1) % NPTR;
    ptrs[k].p = p; ptrs[k].sz = sz; ptrs[k].s = s;
    if (live > hi) { hi = live; if (live >= next_dump) { dump(); next_dump = live + step; } }
    pthread_mutex_unlock(&mu);
    guard = 0;
}
static void untrack(void* p) {
    if (!p || guard) return;
    guard = 1;
    pthread_mutex_lock(&mu);
    uint64_t k = ((uint64_t)p >> 4) % NPTR;
    for (uint64_t c = 0; c < NPTR && ptrs[k].p; c++, k = (k + 1) % NPTR) {
        if (ptrs[k].p == p) {
            stks[ptrs[k].s].live -= ptrs[k].sz; live -= ptrs[k].sz;
            // backward-shift deletion keeps linear probing chains intact
            uint64_t i = k, j = k;
            for (;;) {
                ptrs[i].p = 0; 
                for (;;) {
                    j = (j + 1) % NPTR;
                    if (!ptrs[j].p) goto done;
                    uint64_t r = ((uint64_t)ptrs[j].p >> 4) % NPTR;
                    if ((i <= j) ? (i < r && r <= j) : (i < r || r <= j)) continue;
                    break;
                }
                ptrs[i] = ptrs[j]; i = j;
            }
        }
    }
done:
    pthread_mutex_unlock(&mu);
    guard = 0;
}
void* malloc(size_t n) {
    if (!r_malloc) { if (guard) { void* p = boot + bootn; bootn += (n + 15) & ~15ul; return p; } guard = 1; init(); guard = 0; }
    void* p = r_malloc(n); track(p, n); return p;
}
void* calloc(size_t a, size_t b) {
    if (!r_calloc) { void* p = boot + bootn; bootn += (a * b + 15) & ~15ul; memset(p, 0, a * b); return p; }
    void* p = r_calloc(a, b); track(p, a * b); return p;
}
void free(void* p) {
    if ((char*)p >= boot && (char*)p < boot + sizeof(boot)) return;
    if (!r_free) init();
    untrack(p); r_free(p);
}
void* realloc(void* p, size_t n) {
    if (!r_realloc) init();
    if ((char*)p >= boot && (char*)p < boot + sizeof(boot)) { void* q = r_malloc(n); memcpy(q, p, n); track(q, n); return q; }
    untrack(p); void* q = r_realloc(p, n); track(q, n); return q;
}
void* mmap(void* a, size_t n, int pr, int fl, int fd, off_t o) {
    if (!r_mmap) init();
    void* p = r_mmap(a, n, pr, fl, fd, o);
    if (p != MAP_FAILED && (fl & MAP_ANONYMOUS) && !guard && n >= (1 << 20)) track(p, n);
    return p;
}
int munmap(void* a, size_t n) { if (!r_munmap) init(); untrack(a); return r_munmap(a, n); }
