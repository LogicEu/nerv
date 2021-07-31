
/*********************************************
 *       matrix functions and operations
 * ******************************************/

#define MATRIX_AT(m, x, y) (m->data + (m->columns * y) + x)

#include <nerv.h>
#include <stdio.h>

Mat matrix_scale(Mat* mat, float scale)
{
    Mat ret = matrix_copy(mat);
    int size = ret.rows * ret.columns;

    float* f = ret.data;
    for (float* end = f + size; f != end; f++) {
        *f *= scale;
    }

    return ret;
}

Mat matrix_hadamard(Mat* a, Mat* b)
{
    Mat ret = matrix(a->rows, a->columns);
    if ((a->rows != b->rows) || (a->columns != b->columns)) {
        printf("Matrixes are not equal for Hadamard Product\n");
        return ret;
    }

    int size = ret.rows * ret.columns;
    float* f = ret.data, *af = a->data, *bf = a->data;
    for (float* end = f + size; f != end; f++) {
        *f = (*(af++)) * (*(bf++));
    }

    return ret;
}

Mat matrix_transpose(Mat* m)
{
    Mat ret = matrix(m->columns, m->rows); Mat* r = &ret;
    
    float *f = m->data;
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->columns; x++) {
            *MATRIX_AT(r, y, x) = *(f++);
        }
    }

    return ret;
}

Mat matrix_multiply(Mat* a, Mat* b)
{
    Mat ret = matrix(a->rows, b->columns);
    if (a->columns != b->rows) {
        printf("Number of columns in first matrix must be equal to number of rows in second\n");
        return ret;
    }

    float* f = ret.data;
    for (int y = 0; y < ret.rows; y++) {
        for (int x = 0; x < ret.columns; x++) {
            for (int z = 0; z < a->columns; z++) {
                *f += (*MATRIX_AT(a, z, y)) * (*MATRIX_AT(b, x, z));
            }
            f++;
        }
    }
    
    return ret;
}