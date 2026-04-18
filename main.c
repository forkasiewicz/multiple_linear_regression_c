#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
} params;

mem_arena *arena_create(u64 size);
void arena_destroy(mem_arena *arena);
void *arena_alloc(mem_arena *arena, u64 size);
void arena_free(mem_arena *arena, u64 size);
u64 arena_mark(mem_arena *arena);
void arena_goto(mem_arena *arena, u64 mark);

mat *mat_create(mem_arena *restrict arena, u32 rows, u32 cols, f32 *init);
mat *mat_mul(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b);
mat *mat_transpose(mem_arena *arena, mat *restrict mat_a);
mat *mat_scale(mem_arena *arena, mat *restrict mat_a, f32 num);
mat *mat_sum_float(mem_arena *arena, mat *restrict mat_a, f32 num);
mat *mat_sum_mat(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b);
mat *mat_sub_float(mem_arena *arena, mat *restrict mat_a, f32 num);
mat *mat_sub_mat(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b);

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

u64 arena_mark(mem_arena *arena) { return arena->offset; }
void arena_goto(mem_arena *arena, u64 mark) { arena->offset = mark; }

mat *mat_create(mem_arena *restrict arena, u32 rows, u32 cols, f32 *init) {
  mat *matrix = arena_alloc(arena, sizeof(mat) + sizeof(f32) * rows * cols);
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->data = (f32 *)(matrix + 1);

  for (u32 i = 0; i < rows * cols; i++) {
    matrix->data[i] = init ? init[i] : 0.0f;
  }

  return matrix;
}

mat *mat_mul(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b) {
  mat *mat_c = mat_create(arena, mat_a->rows, mat_b->cols, 0);

  for (u32 i = 0; i < mat_c->rows * mat_c->cols; i++) {
    mat_c->data[i] = 0.0f;
  }

  for (u32 i = 0; i < mat_a->rows; i++) {
    for (u32 j = 0; j < mat_b->cols; j++) {
      for (u32 k = 0; k < mat_a->cols; k++) {
        mat_c->data[i * mat_c->cols + j] +=
            mat_a->data[i * mat_a->cols + k] * mat_b->data[k * mat_b->cols + j];
      }
    }
  }

  return mat_c;
}

mat *mat_transpose(mem_arena *arena, mat *restrict mat_a) {
  mat *mat_t = mat_create(arena, mat_a->cols, mat_a->rows, 0);
  for (u32 i = 0; i < mat_t->rows; i++) {
    for (u32 j = 0; j < mat_t->cols; j++) {
      mat_t->data[i * mat_t->cols + j] = mat_a->data[j * mat_a->cols + i];
    }
  }
  return mat_t;
}

mat *mat_scale(mem_arena *arena, mat *restrict mat_a, f32 f) {
  mat *mat_b = mat_create(arena, mat_a->rows, mat_a->cols, NULL);
  for (u32 i = 0; i < mat_a->rows * mat_a->cols; i++) {
    mat_b->data[i] = mat_a->data[i] * f;
  }
  return mat_b;
}

f32 mat_sum(mat *restrict matrix) {
  f32 sum = 0.0f;
  for (u32 i = 0; i < matrix->cols * matrix->rows; i++) {
    sum += matrix->data[i];
  }
  return sum;
}

mat *mat_sum_float(mem_arena *arena, mat *restrict mat_a, f32 f) {
  mat *mat_b = mat_create(arena, mat_a->rows, mat_a->cols, NULL);
  for (u32 i = 0; i < mat_a->rows * mat_a->cols; i++) {
    mat_b->data[i] = mat_a->data[i] + f;
  }
  return mat_b;
}

mat *mat_sum_mat(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b) {
  mat *mat_c = mat_create(arena, mat_a->rows, mat_a->cols, 0);
  for (u32 i = 0; i < mat_c->rows * mat_c->cols; i++) {
    mat_c->data[i] = mat_a->data[i] + mat_b->data[i];
  }
  return mat_c;
}

mat *mat_sub_float(mem_arena *arena, mat *restrict mat_a, f32 f) {
  mat *mat_b = mat_create(arena, mat_a->rows, mat_a->cols, NULL);
  for (u32 i = 0; i < mat_a->rows * mat_a->cols; i++) {
    mat_b->data[i] = mat_a->data[i] - f;
  }
  return mat_b;
}

mat *mat_sub_mat(mem_arena *arena, mat *restrict mat_a, mat *restrict mat_b) {
  mat *mat_c = mat_create(arena, mat_a->rows, mat_a->cols, 0);
  for (u32 i = 0; i < mat_c->rows * mat_c->cols; i++) {
    mat_c->data[i] = mat_a->data[i] - mat_b->data[i];
  }
  return mat_c;
}

void black_box(mem_arena *arena, params *p) {
  u32 m = p->X->rows;
  mat *y_hat;

  u32 iterations = 1000;
  f32 learning_step = 0.01f;

  mat *X_transposed = mat_transpose(arena, p->X);

  for (u32 i = 0; i < iterations; i++) {
    u64 mark = arena_mark(arena);

    y_hat = mat_sum_float(arena, mat_mul(arena, p->X, p->w), p->b);

    mat *error = mat_sub_mat(arena, y_hat, p->y);

    mat *dw = mat_scale(arena, mat_mul(arena, X_transposed, error), 1.0f / m);

    f32 db = (1.0f / m) * mat_sum(error);

    for (u32 j = 0; j < p->w->rows * p->w->cols; j++) {
      p->w->data[j] -= learning_step * dw->data[j];
    }

    p->b -= db * learning_step;

    arena_goto(arena, mark);
  }
}

i32 main(void) {
  mem_arena *arena = arena_create(MiB(20));

  params *p = arena_alloc(arena, sizeof(params));

  f32 test_X[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                  2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 3.0f};

  f32 test_y[] = {1.0f, 4.0f, 3.0f, 6.0f, 10.0f, 8.0f, 14.0f, 13.0f};

  p->X = mat_create(arena, 8, 2, test_X);
  p->y = mat_create(arena, 8, 1, test_y);
  p->w = mat_create(arena, 2, 1, NULL);
  p->b = 0.0f;

  black_box(arena, p);

  for (u32 i = 0; i < p->w->cols * p->w->rows; i++) {
    if (i % p->w->cols == 0) {
      printf("\n");
    }
    printf("%.0f ", p->w->data[i]);
  }
  printf("\n");

  arena_destroy(arena);
  return EXIT_SUCCESS;
}
