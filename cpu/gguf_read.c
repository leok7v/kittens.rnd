// gguf_read.c -- GGUF v3 reader implementation. See kt_gguf.h.

#include "kt_gguf.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


// ---------------------------------------------------------------------------
// GGUF on-disk format (v3) summary
// ---------------------------------------------------------------------------
//
// uint32_t  magic        = 'G''G''U''F' = 0x46554747 LE
// uint32_t  version      = 3
// uint64_t  n_tensors
// uint64_t  n_kv
// kv[n_kv]                                  // key/value metadata
// ti[n_tensors]                             // tensor info: name, ne[], type, off
// (pad to general.alignment, default 32)
// uint8_t   data[]                          // tensor data section
//
// Each KV:
//   uint64_t key_len; char key[key_len];
//   uint32_t type;
//   value (depends on type; ARRAY adds nested type + count + items)
//
// Each tensor info:
//   uint64_t name_len; char name[name_len];
//   uint32_t n_dims;
//   uint64_t ne[n_dims];                    // shape, innermost first
//   uint32_t dtype;                          // ggml_type
//   uint64_t offset;                         // relative to data section

#define GGUF_MAGIC    0x46554747u
#define GGUF_VERSION  3

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum ggml_type_min {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
};


// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

typedef struct {
    char *  key;          // owned (strndup'd from mmap)
    int     vtype;        // gguf_type
    // Scalar value union (used for u32 / f32 / bool / string only).
    uint32_t u32;
    float    f32;
    int      bv;
    char *   str;         // owned for STRING; NULL otherwise
} kt_gguf_kv;

typedef struct {
    char *   name;        // owned
    int32_t  dtype;       // ggml_type (we only read F32/F16)
    int      ndim;
    int64_t  ne[4];
    size_t   offset;      // from start of data section
    size_t   nbytes;
} kt_gguf_ti;

struct kt_gguf {
    int          fd;
    const char * map_base;      // const view into the mmap
    size_t       map_size;
    size_t       data_off;      // absolute byte offset of data section

    kt_gguf_kv * kv;
    int          n_kv;
    kt_gguf_ti * ti;
    int          n_ti;
};


// ---------------------------------------------------------------------------
// Error reporting (single TLS-less slot; this loader is not thread-safe)
// ---------------------------------------------------------------------------

static char g_err[256] = "";

static void set_err(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

const char * kt_gguf_last_error(void) {
    return g_err;
}


// ---------------------------------------------------------------------------
// Byte-stream cursor over the mmap region
// ---------------------------------------------------------------------------

typedef struct {
    const char * base;
    size_t       size;
    size_t       pos;
    int          ok;          // 0 = ran off end, 1 = valid
} kt_cur;

static void cur_init(kt_cur * c, const char * base, size_t size) {
    c->base = base; c->size = size; c->pos = 0; c->ok = 1;
}
static void cur_read(kt_cur * c, void * dst, size_t n) {
    if (c->pos + n > c->size) { c->ok = 0; return; }
    memcpy(dst, c->base + c->pos, n);
    c->pos += n;
}
static void cur_skip(kt_cur * c, size_t n) {
    if (c->pos + n > c->size) { c->ok = 0; return; }
    c->pos += n;
}
static uint32_t cur_u32(kt_cur * c) {
    uint32_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}
static uint64_t cur_u64(kt_cur * c) {
    uint64_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}
static int32_t cur_i32(kt_cur * c) {
    int32_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}
__attribute__((unused))
static int64_t cur_i64(kt_cur * c) {
    int64_t v = 0; cur_read(c, &v, sizeof(v)); return v;
}
static float cur_f32(kt_cur * c) {
    float v = 0; cur_read(c, &v, sizeof(v)); return v;
}
__attribute__((unused))
static double cur_f64(kt_cur * c) {
    double v = 0; cur_read(c, &v, sizeof(v)); return v;
}
static char * cur_str_dup(kt_cur * c) {
    uint64_t n = cur_u64(c);
    char * s = NULL;
    if (c->ok && c->pos + n <= c->size) {
        s = (char *)malloc(n + 1);
        if (s != NULL) {
            memcpy(s, c->base + c->pos, n);
            s[n] = '\0';
        }
        c->pos += n;
    } else {
        c->ok = 0;
    }
    return s;
}

// Size in bytes of a gguf scalar value (no length prefix). For STRING
// the length is part of the value; we return the actual size by
// peeking. The cursor is NOT advanced.
static int gguf_scalar_size(int vtype, kt_cur * peek_cur,
                            size_t * out_bytes) {
    int ok = 1;
    size_t bytes = 0;
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    bytes = 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   bytes = 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: bytes = 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: bytes = 8; break;
        case GGUF_TYPE_STRING: {
            // length-prefixed; need the u64 + body.
            if (peek_cur->pos + 8 > peek_cur->size) { ok = 0; break; }
            uint64_t n = 0;
            memcpy(&n, peek_cur->base + peek_cur->pos, 8);
            bytes = 8 + n;
            break;
        }
        default: ok = 0; break;
    }
    *out_bytes = bytes;
    return ok;
}


// ---------------------------------------------------------------------------
// File open / close
// ---------------------------------------------------------------------------

static int kt_gguf_parse(kt_gguf * g) {
    int ok = 0;
    kt_cur c; cur_init(&c, g->map_base, g->map_size);
    uint32_t magic   = cur_u32(&c);
    uint32_t version = cur_u32(&c);
    uint64_t n_ti    = cur_u64(&c);
    uint64_t n_kv    = cur_u64(&c);
    if (!c.ok || magic != GGUF_MAGIC || version != GGUF_VERSION) {
        set_err("bad GGUF magic/version (got %08x v%u)",
                magic, version);
    } else if (n_ti > 100000 || n_kv > 10000) {
        set_err("absurd GGUF counts (%llu tensors, %llu kvs)",
                (unsigned long long)n_ti, (unsigned long long)n_kv);
    } else {
        g->n_kv = (int)n_kv;
        g->n_ti = (int)n_ti;
        g->kv   = (kt_gguf_kv *)calloc((size_t)g->n_kv,
                                       sizeof(kt_gguf_kv));
        g->ti   = (kt_gguf_ti *)calloc((size_t)g->n_ti,
                                       sizeof(kt_gguf_ti));
        // KVs.
        int kv_ok = 1;
        for (int i = 0; i < g->n_kv && kv_ok; i++) {
            kt_gguf_kv * kv = &g->kv[i];
            kv->key = cur_str_dup(&c);
            kv->vtype = (int)cur_u32(&c);
            if (!c.ok || kv->key == NULL) {
                kv_ok = 0;
                break;
            }
            switch (kv->vtype) {
                case GGUF_TYPE_UINT32: kv->u32 = cur_u32(&c); break;
                case GGUF_TYPE_INT32:  kv->u32 = (uint32_t)cur_i32(&c); break;
                case GGUF_TYPE_FLOAT32:kv->f32 = cur_f32(&c); break;
                case GGUF_TYPE_BOOL: {
                    uint8_t b = 0; cur_read(&c, &b, 1);
                    kv->bv = b ? 1 : 0; break;
                }
                case GGUF_TYPE_STRING: kv->str = cur_str_dup(&c); break;
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:   cur_skip(&c, 1); break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:  cur_skip(&c, 2); break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:  cur_skip(&c, 8); break;
                case GGUF_TYPE_FLOAT64:cur_skip(&c, 8); break;
                case GGUF_TYPE_ARRAY: {
                    // [u32 inner_type][u64 count][items...]
                    uint32_t inner = cur_u32(&c);
                    uint64_t count = cur_u64(&c);
                    if (inner == GGUF_TYPE_STRING) {
                        for (uint64_t k = 0; k < count && c.ok; k++) {
                            char * s = cur_str_dup(&c);
                            free(s);
                        }
                    } else {
                        size_t per = 0;
                        kt_cur peek = c;
                        if (!gguf_scalar_size((int)inner, &peek, &per)) {
                            kv_ok = 0;
                            break;
                        }
                        cur_skip(&c, per * count);
                    }
                    break;
                }
                default: kv_ok = 0; break;
            }
            if (!c.ok) { kv_ok = 0; }
        }
        if (!kv_ok) {
            set_err("KV parse failed (idx ~%d)", g->n_kv);
        } else {
            // Tensor infos.
            int ti_ok = 1;
            for (int i = 0; i < g->n_ti && ti_ok; i++) {
                kt_gguf_ti * t = &g->ti[i];
                t->name = cur_str_dup(&c);
                uint32_t n_dims = cur_u32(&c);
                if (!c.ok || t->name == NULL || n_dims > 4) {
                    ti_ok = 0; break;
                }
                t->ndim = (int)n_dims;
                for (int d = 0; d < 4; d++) { t->ne[d] = 1; }
                for (uint32_t d = 0; d < n_dims; d++) {
                    t->ne[d] = (int64_t)cur_u64(&c);
                }
                t->dtype  = cur_i32(&c);
                t->offset = (size_t)cur_u64(&c);
                if (!c.ok) { ti_ok = 0; }
            }
            if (!ti_ok) {
                set_err("tensor info parse failed");
            } else {
                // Align to general.alignment (default 32).
                uint32_t align = 32;
                for (int i = 0; i < g->n_kv; i++) {
                    if (strcmp(g->kv[i].key, "general.alignment") == 0
                        && g->kv[i].vtype == GGUF_TYPE_UINT32) {
                        align = g->kv[i].u32;
                    }
                }
                size_t mask = (size_t)align - 1;
                g->data_off = (c.pos + mask) & ~mask;
                // Compute per-tensor nbytes from ne + dtype.
                for (int i = 0; i < g->n_ti; i++) {
                    kt_gguf_ti * t = &g->ti[i];
                    int64_t nel = 1;
                    for (int d = 0; d < t->ndim; d++) { nel *= t->ne[d]; }
                    size_t per = (t->dtype == GGML_TYPE_F32) ? 4u
                               : (t->dtype == GGML_TYPE_F16) ? 2u : 0u;
                    if (per == 0) {
                        set_err("tensor %s: unsupported dtype %d",
                                t->name, t->dtype);
                        ti_ok = 0;
                        break;
                    }
                    t->nbytes = (size_t)nel * per;
                }
                if (ti_ok) { ok = 1; }
            }
        }
    }
    return ok;
}

kt_gguf * kt_gguf_open(const char * path) {
    kt_gguf * result = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err("open(%s): errno %d", path, errno);
    } else {
        struct stat st;
        if (fstat(fd, &st) != 0) {
            set_err("fstat: errno %d", errno);
            close(fd);
        } else {
            size_t size = (size_t)st.st_size;
            void * m = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) {
                set_err("mmap: errno %d", errno);
                close(fd);
            } else {
                kt_gguf * g = (kt_gguf *)calloc(1, sizeof(kt_gguf));
                g->fd = fd;
                g->map_base = (const char *)m;
                g->map_size = size;
                if (kt_gguf_parse(g)) {
                    result = g;
                } else {
                    munmap(m, size);
                    close(fd);
                    free(g->kv);
                    free(g->ti);
                    free(g);
                }
            }
        }
    }
    return result;
}

void kt_gguf_close(kt_gguf * g) {
    if (g != NULL) {
        for (int i = 0; i < g->n_kv; i++) {
            free(g->kv[i].key);
            free(g->kv[i].str);
        }
        free(g->kv);
        for (int i = 0; i < g->n_ti; i++) { free(g->ti[i].name); }
        free(g->ti);
        if (g->map_base != NULL) {
            munmap((void *)g->map_base, g->map_size);
        }
        if (g->fd >= 0) { close(g->fd); }
        free(g);
    }
}


// ---------------------------------------------------------------------------
// KV lookups
// ---------------------------------------------------------------------------

static const kt_gguf_kv * kt_gguf_find_kv(const kt_gguf * g,
                                          const char * key,
                                          int vtype) {
    const kt_gguf_kv * found = NULL;
    for (int i = 0; i < g->n_kv; i++) {
        if (g->kv[i].vtype == vtype
            && strcmp(g->kv[i].key, key) == 0) {
            found = &g->kv[i];
        }
    }
    return found;
}

int kt_gguf_get_u32(const kt_gguf * g, const char * key,
                    uint32_t * out) {
    const kt_gguf_kv * kv = kt_gguf_find_kv(g, key, GGUF_TYPE_UINT32);
    int got = 0;
    if (kv != NULL) { *out = kv->u32; got = 1; }
    return got;
}

int kt_gguf_get_f32(const kt_gguf * g, const char * key, float * out) {
    const kt_gguf_kv * kv = kt_gguf_find_kv(g, key, GGUF_TYPE_FLOAT32);
    int got = 0;
    if (kv != NULL) { *out = kv->f32; got = 1; }
    return got;
}

int kt_gguf_get_bool(const kt_gguf * g, const char * key, int * out) {
    const kt_gguf_kv * kv = kt_gguf_find_kv(g, key, GGUF_TYPE_BOOL);
    int got = 0;
    if (kv != NULL) { *out = kv->bv; got = 1; }
    return got;
}


// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

int kt_gguf_n_tensors(const kt_gguf * g) { return g->n_ti; }

const char * kt_gguf_tensor_name(const kt_gguf * g, int i) {
    assert(i >= 0 && i < g->n_ti);
    return g->ti[i].name;
}

static const kt_gguf_ti * find_tensor_info(const kt_gguf * g,
                                           const char * name) {
    const kt_gguf_ti * found = NULL;
    for (int i = 0; i < g->n_ti; i++) {
        if (strcmp(g->ti[i].name, name) == 0) { found = &g->ti[i]; }
    }
    return found;
}

kt_tensor * kt_gguf_load_tensor(const kt_gguf * g, kt_arena * arena,
                                const char * name) {
    const kt_gguf_ti * ti = find_tensor_info(g, name);
    kt_tensor * out = NULL;
    if (ti != NULL) {
        const char * src = g->map_base + g->data_off + ti->offset;
        int64_t n0 = ti->ne[0];
        int64_t n1 = ti->ne[1];
        int64_t n2 = ti->ne[2];
        int64_t n3 = ti->ne[3];
        if (ti->dtype == GGML_TYPE_F32) {
            // Wrap the mmap'd bytes directly (zero-copy). The arena
            // owns only the tensor header; the data lives in the file.
            if (ti->ndim == 1) {
                out = kt_wrap_2d(arena, (float *)src, n0, 1);
                out->ndim = 1;
            } else if (ti->ndim == 2) {
                out = kt_wrap_2d(arena, (float *)src, n0, n1);
            } else if (ti->ndim == 3) {
                out = kt_wrap_3d(arena, (float *)src, n0, n1, n2);
            } else {
                // 4D: not used by kittens-tts; fall through if needed.
                out = kt_wrap_3d(arena, (float *)src, n0, n1, n2);
                out->ndim = 4;
                out->ne[3] = n3;
                out->nb[3] = out->nb[2] * n2;
            }
        } else if (ti->dtype == GGML_TYPE_F16) {
            // Dequant on load: F16 -> F32 into a fresh arena buffer.
            kt_tensor * t = NULL;
            if (ti->ndim == 1) { t = kt_new_1d(arena, n0); }
            else if (ti->ndim == 2) { t = kt_new_2d(arena, n0, n1); }
            else if (ti->ndim == 3) { t = kt_new_3d(arena, n0, n1, n2); }
            else                    { t = kt_new_4d(arena, n0, n1, n2, n3); }
            int64_t total = kt_nelements(t);
            const uint16_t * src16 = (const uint16_t *)src;
            float * dst = t->data;
            for (int64_t i = 0; i < total; i++) {
                _Float16 h;
                memcpy(&h, &src16[i], 2);
                dst[i] = (float)h;
            }
            out = t;
        } else {
            assert(0 && "kt_gguf: unsupported tensor dtype");
        }
        if (out != NULL) { kt_set_name(out, name); }
    }
    return out;
}
