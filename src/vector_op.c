
/*********************************************
 *     vector functions and operations
 * ******************************************/

#include <nerv.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void vector_add(const Vec* restrict dst, const Vec* restrict src)
{
    if (dst->size != src->size) {
        printf("Add: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }

    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++, n++) {
        *f += *n;
    }
}

void vector_sub(const Vec* restrict dst, const Vec* restrict src)
{
    if (dst->size != src->size) {
        printf("Sub: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }
    
    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++, n++) {
        *f -= *n;
    }
}

void vector_hadamard(const Vec* restrict dst, const Vec* restrict src)
{
    if (dst->size != src->size) {
        printf("Hadamard: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }

    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++, n++) {
        *f *= *n;
    }
}

void vector_scale(const Vec* restrict v, float n)
{
    float *f = v->data;
    for (float* end = f + v->size; f != end; f++) {
        *f *= n;
    }
}

Vec vector_by_matrix(const Mat* restrict mat, const Vec* restrict vec)
{
    Vec ret = vector(mat->rows);
    if (vec->size != mat->columns) {
        printf("Vector By Matrix: Vector size must be equal to matrix columns\n");
        return ret;
    }

    float* f = ret.data, *m = mat->data;
    for (int y = 0; y < mat->rows; y++, f++) {
        float* n = vec->data;
        for (int x = 0; x < mat->columns; x++, m++, n++) {
            *f += (*m) * (*n);
        }
    }

    return ret;
}

Vec vector_by_matrix_transposed(const Mat* restrict mat, const Vec* restrict vec)
{
    Mat m;
    memcpy(&m, mat, sizeof(Mat));

    m.columns = m.rows;
    m.rows = mat->columns;
    
    return vector_by_matrix(&m, vec);
}

Vec vector_relu(const Vec* restrict v)
{
    Vec ret = vector(v->size);
    
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++, n++) {
        *r = _relu(*n);
    }

    return ret;
}

Vec vector_drelu(const Vec* restrict v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++, n++) {
        *r = _drelu(*n);
    }
    return ret;
}

Vec vector_sigmoid(const Vec* restrict v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++, n++) {
        *r = _sigmoid(*n);
    }
    return ret;
}

Vec vector_dsigmoid(const Vec* restrict v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++, n++) {
        *r = _dsigmoid(*n);
    }
    return ret;
}

Vec vector_sigderiv(const Vec* restrict v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++, n++) {
        *r = _sigderiv(*n);
    }
    return ret;
}

Vec vector_leaky_relu(const Vec* restrict v, float leak)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = leaky_relu(*(n++), leak);
    }
    return ret;
}

Vec vector_dleaky_relu(const Vec* restrict v, float leak)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = dleaky_relu(*(n++), leak);
    }
    return ret;
}

Vec vector_softmax(const Vec* restrict v)
{
    float* n = v->data, total = 0.0f, *f;
    for (float* end = n + v->size; n != end; n++) {
        total += *n;
    }

    Vec ret = vector(v->size);
    n = ret.data, f = v->data;
    for (float* end = n + ret.size; n != end; n++) {
        *n = *(f++) / total;
    }
    return ret;
}