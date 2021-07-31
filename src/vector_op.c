
/*********************************************
 *     vector functions and operations
 * ******************************************/

#include <nerv.h>
#include <stdio.h>
#include <string.h>

void vector_add(Vec* dst, Vec* src)
{
    if (dst->size != src->size) {
        printf("Add: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }

    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++) {
        *f += *(n++);
    }
}

void vector_sub(Vec* dst, Vec* src)
{
    if (dst->size != src->size) {
        printf("Sub: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }
    
    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++) {
        *f -= *(n++);
    }
}

void vector_hadamard(Vec* dst, Vec* src)
{
    if (dst->size != src->size) {
        printf("Hadamard: Vector A (%d) and B (%d) are not the same size\n", dst->size, src->size);
        return;
    }

    float* f = dst->data, *n = src->data;
    for (float* end = f + dst->size; f != end; f++) {
        *f *= *(n++);
    }
}

void vector_scale(Vec* v, float n)
{
    float *f = v->data;
    for (float* end = f + v->size; f != end; f++) {
        *f *= n;
    }
}

Vec vector_by_matrix(Mat* mat, Vec* vec)
{
    Vec ret = vector(mat->rows);
    if (vec->size != mat->columns) {
        printf("Vector By Matrix: Vector size must be equal to matrix columns\n");
        return ret;
    }

    float* f = ret.data, *m = mat->data;
    for (int y = 0; y < mat->rows; y++) {
        float* n = vec->data;
        for (int x = 0; x < mat->columns; x++) {
            *f += (*m) * (*n);
            m++;
            n++;
        }
        f++;
    }

    return ret;
}

Vec vector_by_matrix_transposed(Mat* mat, Vec* vec)
{
    Mat m;
    memcpy(&m, mat, sizeof(Mat));

    m.columns = m.rows;
    m.rows = mat->columns;
    
    return vector_by_matrix(&m, vec);
}

Vec vector_relu(Vec* v)
{
    Vec ret = vector(v->size);
    
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = relu(*(n++));
    }

    return ret;
}

Vec vector_drelu(Vec* v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = drelu(*(n++));
    }
    return ret;
}

Vec vector_sigmoid(Vec* v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = sigmoid(*(n++));
    }
    return ret;
}

Vec vector_dsigmoid(Vec* v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = dsigmoid(*(n++));
    }
    return ret;
}

Vec vector_sigderiv(Vec* v)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = sigderiv(*(n++));
    }
    return ret;
}

Vec vector_leaky_relu(Vec* v, float leak)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = leaky_relu(*(n++), leak);
    }
    return ret;
}

Vec vector_dleaky_relu(Vec* v, float leak)
{
    Vec ret = vector(v->size);
    float* r = ret.data, *n = v->data;
    for (float* end = r + ret.size; r != end; r++) {
        *r = dleaky_relu(*(n++), leak);
    }
    return ret;
}

Vec vector_softmax(Vec* v)
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