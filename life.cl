__kernel void life_next_state(uint rows, uint cols, __global uchar *cells, __global uchar *next)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i >= rows || j >= cols) {
		return;
	}

	uint idx = i * cols + j;
	uint neighbors = 0;
	for (int di = -1; di <= 1; di += 1) {
		for (int dj = -1; dj <= 1; dj += 1) {
			int ii = (i+di) % rows, jj = (j+dj) % cols;

			neighbors += cells[ii * cols + jj];
		}
	}
	uchar alive = cells[idx];
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
	next[idx] = alive;
}

// vim: ft=c
