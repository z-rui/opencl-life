#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <CL/cl.h>

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

typedef struct {
	size_t rows, cols;
	uint8_t *cells;
} life_state;

void init_life_state(life_state *s, size_t rows, size_t cols)
{
	s->rows = rows;
	s->cols = cols;
	s->cells = calloc(rows * cols, 1);
}

void deinit_life_state(life_state *s)
{
	free(s->cells);
}

typedef struct {
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem in_buffer, out_buffer;
} accel_ctx;

void life_next_state(life_state *s, accel_ctx *acc)
{
	cl_int err;
	size_t global_size[2] = {s->rows, s->cols};

	err = clEnqueueWriteBuffer(acc->queue, acc->in_buffer, CL_FALSE, 0, s->rows * s->cols, s->cells, 0, NULL, NULL);
	if (err < 0) {
		printf("couldn't write buffer: %d\n", err);
		exit(1);
	}
	err = clEnqueueNDRangeKernel(acc->queue, acc->kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
	if (err < 0) {
		printf("couldn't enqueue kernel: %d\n", err);
		exit(1);
	}
	err = clEnqueueReadBuffer(acc->queue, acc->out_buffer, CL_TRUE, 0, s->rows * s->cols, s->cells, 0, NULL, NULL);
	if (err < 0) {
		printf("couldn't read buffer: %d\n", err);
		exit(1);
	}
}

void life_next_state_dumb(life_state *s)
{
	uint8_t *next;
	size_t idx = 0;

	next = malloc(s->rows * s->cols);
	for (int i = 0; i < s->rows; i++) {
		for (int j = 0; j < s->cols; j++) {
			int neighbors = 0;
			for (int di = -1; di <= 1; di += 1) {
				for (int dj = -1; dj <= 1; dj += 1) {
					int ii = (i+di) % s->rows, jj = (j+dj) % s->cols;

					neighbors += s->cells[ii * s->cols + jj];
				}
			}
			uint8_t alive = s->cells[idx];
			if (alive == 0) {
				if (neighbors == 3) {
					alive = 1;
				}
			} else {
				// neighbor includes itself.
				// requires 2-3 neighbors excluding itself
				if (neighbors != 3 && neighbors != 4) {
					alive = 0;
				}
			}
			next[idx++] = alive;
		}
	}
	free(s->cells);
	s->cells = next;
}

void show_life_state(life_state *s)
{
	size_t idx = 0;

	putchar('\n');
	for (size_t i = 0; i < s->rows; i++) {
		for (size_t j = 0; j < s->cols; j++) {
			putchar(s->cells[idx++] ? '*' : ' ');
		}
		putchar('\n');
	}
}

void rand_life_state(life_state *s)
{
	size_t idx = 0;
	for (size_t i = 0; i < s->rows; i++) {
		for (size_t j = 0; j < s->cols; j++) {
			s->cells[idx++] = rand() % 2;
		}
	}
}

#define PROGRAM_FILE "life.cl"
#define KERNEL_NAME "life_next_state"

int main()
{
	int rows = 2160, cols = 3840;

	cl_int err = 0;
	cl_device_id dev = create_device();
	cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	cl_program program = build_program(ctx, dev, PROGRAM_FILE);
	accel_ctx acc;

	acc.in_buffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY, rows * cols, NULL, &err);
	acc.out_buffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, rows * cols, NULL, &err);
	if (err < 0) {
		printf("couldn't create buffer: %d\n", err);
		exit(1);
	}
	acc.queue = clCreateCommandQueue(ctx, dev, 0, &err);
	if (err < 0) {
		printf("couldn't create command queue: %d\n", err);
		exit(1);
	}
	acc.kernel = clCreateKernel(program, KERNEL_NAME, &err);
	if (err < 0) {
		printf("couldn't create kernel: %d\n", err);
		exit(1);
	}
	cl_uint n = rows, m = cols;
	err |= clSetKernelArg(acc.kernel, 0, sizeof n, &n);
	err |= clSetKernelArg(acc.kernel, 1, sizeof m, &m);
	err |= clSetKernelArg(acc.kernel, 2, sizeof (cl_mem), &acc.in_buffer);
	err |= clSetKernelArg(acc.kernel, 3, sizeof (cl_mem), &acc.out_buffer);
	if (err < 0) {
		printf("couldn't set kernel args: %d\n", err);
		exit(1);
	}

	life_state lstate;

	init_life_state(&lstate, rows, cols);
	rand_life_state(&lstate);

#define ITERATIONS 200
	clock_t start = clock();
	for (int i = 0; i < ITERATIONS; i++) {
		//show_life_state(&lstate);
		//getchar();
		life_next_state(&lstate, &acc);
		//life_next_state_dumb(&lstate);
	}
	clock_t end = clock();
	printf("time elapsed: %lfs   (%dx%d, %d iterations)\n", (double) (end-start) / CLOCKS_PER_SEC, rows, cols, ITERATIONS);

	deinit_life_state(&lstate);
	clReleaseKernel(acc.kernel);
	clReleaseCommandQueue(acc.queue);
	clReleaseMemObject(acc.in_buffer);
	clReleaseMemObject(acc.out_buffer);
	clReleaseProgram(program);
	clReleaseContext(ctx);
}
