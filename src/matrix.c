
/*********************************************
 *       matrix creation and management
 * ******************************************/

#include <nerv.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

Mat matrix(int rows, int columns)
{
    Mat mat;
    mat.rows = rows;
    mat.columns = columns;
    mat.data = (float*)calloc(rows * columns, sizeof(float));
    return mat;
}

Mat matrix_identity(int size)
{
    Mat ret = matrix(size, size), *m; m = &ret;
    float* f = ret.data;
    for (int i = 0; i < size; i++) {
        *f = 1.0f;
        f += size + 1;
    }
    return ret;
}

Mat matrix_create(int rows, int columns, ...)
{
    Mat m = matrix(rows, columns);
    columns *= rows;
    float* f = m.data;

    va_list args;
    va_start(args, columns);
    for (float* end = f + columns; f != end; f++) {
        *f = (float)va_arg(args, double);
    }
    va_end(args);

    return m;
}

Mat matrix_uniform(int rows, int columns, float val)
{
    Mat m = matrix(rows, columns);
    int size = rows * columns;
    
    float* f = m.data;
    for (float* end = f + size; f != end; f++) {
        *f = val;
    }

    return m;
}

Mat matrix_copy(Mat* mat)
{
    Mat ret = matrix(mat->rows, mat->columns);
    memcpy(ret.data, mat->data, sizeof(float) * mat->rows * mat->columns);
    return ret;
}

Mat matrix_vector(Vec* v)
{
    Mat m = matrix(v->size, 1);
    memcpy(m.data, v->data, sizeof(float) * v->size);
    return m;
}

void matrix_free(Mat* mat)
{
    free(mat->data);
}