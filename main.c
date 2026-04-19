#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ASSERT(_e, ...)                                                        \
  if (!(_e)) {                                                                 \
    fprintf(stderr, __VA_ARGS__);                                              \
    exit(EXIT_FAILURE);                                                        \
  }

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MEM_ALIGN(s, p) (((s) + (p) - 1) & ~((p) - 1))
#define ARENA_ALIGNMENT (sizeof(void *))

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef struct {
  u8 *memory;
  u64 size;
  u64 offset;
} mem_arena;

typedef struct {
  u32 cols, rows;
  f32 *data;
} mat;

typedef struct {
  mat *X;
  mat *y;
  mat *w;
  f32 b;
} parameters;

mem_arena *arena_create(u64 size);
void arena_destroy(mem_arena *arena);
void *arena_alloc(mem_arena *arena, u64 size);
void arena_free(mem_arena *arena, u64 size);
u64 arena_mark(mem_arena *arena);
void arena_goto(mem_arena *arena, u64 mark);

mat *mat_create(mem_arena *arena, u32 rows, u32 cols, f32 *init);
void mat_mul(mat *out, mat *a, mat *b);
void mat_transpose(mat *out, mat *a);
void mat_scale(mat *out, mat *a, f32 f);
f32 mat_sum(mat *a);
void mat_sum_float(mat *out, mat *a, f32 f);
void mat_sum_mat(mat *out, mat *a, mat *b);
void mat_sub_float(mat *out, mat *a, f32 f);
void mat_sub_mat(mat *out, mat *a, mat *b);

mem_arena *arena_create(u64 size) {
  mem_arena *arena = (mem_arena *)malloc(sizeof(mem_arena));
  ASSERT(arena, "failed to allocate arena\n");

  arena->memory = (u8 *)malloc(size);
  ASSERT(arena->memory, "failed to allocate arena memory\n");

  arena->size = size;
  arena->offset = 0;
  return arena;
}

void arena_destroy(mem_arena *arena) {
  free(arena->memory);
  free(arena);
}

void *arena_alloc(mem_arena *arena, u64 size) {
  u64 aligned = MEM_ALIGN(size, ARENA_ALIGNMENT);
  ASSERT(arena->offset + aligned <= arena->size, "arena ran out of memory\n");

  void *mem_ptr = arena->memory + arena->offset;
  arena->offset += aligned;
  return mem_ptr;
}

void arena_free(mem_arena *arena, u64 size) {
  u64 aligned = MEM_ALIGN(size, ARENA_ALIGNMENT);
  ASSERT(arena->offset >= aligned, "arena pop out of bounds\n");
  arena->offset -= aligned;
}

u64 arena_mark(mem_arena *arena) { return arena->offset; };
void arena_goto(mem_arena *arena, u64 mark) { arena->offset = mark; };

mat *mat_create(mem_arena *arena, u32 rows, u32 cols, f32 *init) {
  mat *matrix = arena_alloc(arena, sizeof(mat) + sizeof(f32) * rows * cols);
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->data = (f32 *)(matrix + 1);

  for (u32 i = 0; i < rows * cols; i++) {
    matrix->data[i] = init ? init[i] : 0.0f;
  }

  return matrix;
}

void mat_mul(mat *out, mat *a, mat *b) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == b->cols, "incorrect output matrix size!\n");
  ASSERT(a->cols == b->rows, "incorrect matrix multiplication sizes!\n");

  for (u32 i = 0; i < a->rows; i++) {
    for (u32 j = 0; j < b->cols; j++) {
      f32 sum = 0.0f;
      for (u32 k = 0; k < a->cols; k++) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      out->data[i * out->cols + j] = sum;
    }
  }
}

void mat_transpose(mat *out, mat *a) {
  ASSERT(out->rows == a->cols, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->rows, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows; i++) {
    for (u32 j = 0; j < out->cols; j++) {
      out->data[i * out->cols + j] = a->data[j * a->cols + i];
    }
  }
}

void mat_scale(mat *out, mat *a, f32 f) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->cols, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = a->data[i] * f;
  }
}

f32 mat_sum(mat *a) {
  f32 sum = 0.0f;
  for (u32 i = 0; i < a->cols * a->rows; i++) {
    sum += a->data[i];
  }
  return sum;
}

void mat_sum_float(mat *out, mat *a, f32 f) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->cols, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = a->data[i] + f;
  }
}

void mat_sum_mat(mat *out, mat *a, mat *b) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->cols, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = a->data[i] + b->data[i];
  }
}

void mat_sub_float(mat *out, mat *a, f32 f) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->cols, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = a->data[i] - f;
  }
}

void mat_sub_mat(mat *out, mat *a, mat *b) {
  ASSERT(out->rows == a->rows, "incorrect output matrix size!\n");
  ASSERT(out->cols == a->cols, "incorrect output matrix size!\n");

  for (u32 i = 0; i < out->rows * out->cols; i++) {
    out->data[i] = a->data[i] - b->data[i];
  }
}

void black_box(mem_arena *arena, parameters *p) {
  u64 mark = arena_mark(arena);

  u32 m = p->X->rows;
  u32 iterations = 1600;
  f32 learning_step = 0.01f;

  mat *y_hat = mat_create(arena, p->X->rows, p->w->cols, NULL);
  mat *error = mat_create(arena, y_hat->rows, y_hat->cols, NULL);
  mat *dw = mat_create(arena, p->w->rows, p->w->cols, NULL);
  mat *X_T = mat_create(arena, p->X->cols, p->X->rows, NULL);
  mat_transpose(X_T, p->X);

  for (u32 i = 0; i < iterations; i++) {
    mat_mul(y_hat, p->X, p->w);
    mat_sum_float(y_hat, y_hat, p->b);

    mat_sub_mat(error, y_hat, p->y);

    mat_mul(dw, X_T, error);
    mat_scale(dw, dw, 1.0f / m);

    f32 db = (1.0f / m) * mat_sum(error);

    for (u32 j = 0; j < p->w->rows * p->w->cols; j++) {
      p->w->data[j] -= learning_step * dw->data[j];
    }

    p->b -= db * learning_step;
  }

  printf("arena size: %f MiB\n", (f32)arena->offset / MiB(1));

  arena_goto(arena, mark);
}

// def calculate_rmse(self: "LinearRegression", X: np.ndarray,
// y: np.ndarray) -> float:
//     y_hat = self.predict(X)
//
//     return np.sqrt(np.mean((y_hat - y) ** 2))

f32 calculate_r2(mem_arena *arena, parameters *p) {
  u64 mark = arena_mark(arena);
  mat *y_hat = mat_create(arena, p->X->rows, p->w->cols, NULL);
  mat_mul(y_hat, p->X, p->w);
  mat_sum_float(y_hat, y_hat, p->b);

  f32 y_mean = mat_sum(p->y) / p->y->rows;
  f32 ss_res = 0.0f;
  f32 ss_tot = 0.0f;

  for (u32 i = 0; i < p->y->rows; i++) {
    f32 res = p->y->data[i] - y_hat->data[i];
    ss_res += res * res;

    f32 tot = p->y->data[i] - y_mean;
    ss_tot += tot * tot;
  }

  arena_goto(arena, mark);

  if (ss_tot == 0.0f) {
    return 0.0f;
  } else {
    return 1.0f - (ss_res / ss_tot);
  }
}

// TODO:
// - read csv
// - one-hot encoding
// - standardization
//
// - dynamic iterations
// - calculate rmse
// - split black_box into named functions
// - dynamic arena alloc:
//    - read columns in csv
//    - read rows in csv
//    - create arena of columns * rows * sizeof(f32) * 10?

i32 main(void) {
  mem_arena *arena = arena_create(MiB(20));

  parameters *p = arena_alloc(arena, sizeof(parameters));

  f32 test_X[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                  2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 3.0f};

  f32 test_y[] = {1.0f, 4.0f, 3.0f, 6.0f, 10.0f, 8.0f, 14.0f, 13.0f};

  p->X = mat_create(arena, 8, 2, test_X);
  p->y = mat_create(arena, 8, 1, test_y);
  p->w = mat_create(arena, 2, 1, NULL);
  p->b = 0.0f;

  black_box(arena, p);
  f32 r2 = calculate_r2(arena, p);
  printf("r2: %f\n", r2);
  printf(" b: %f\n", p->b);
  for (u32 i = 0; i < p->w->cols * p->w->rows; i++) {
    printf("%f ", p->w->data[i]);
  }
  printf("\n");

  arena_destroy(arena);
  return EXIT_SUCCESS;
}
