
/*********************************************
 *    neural network model operations 
 * ******************************************/

#include <nerv.h>
#include <stdio.h>
#include <string.h>

static void vector_swap(Vec func, Vec* v)
{
    memcpy(v->data, func.data, v->size * sizeof(float));
    vector_free(&func);
}

static Vec vector_by_matrix_plus(Mat* m, Vec* a, Vec* b)
{
    Vec ret = vector_by_matrix(m, a);
    vector_add(&ret, b);
    return ret;
}

static void matrix_minus_vec_by_vec(Mat* m, Vec* a, Vec* b)
{
    if (a->size != m->columns || b->size != m->rows) {
        printf("Vector and matrix are not the same size!\n");
        return;
    }

    float *f = m->data;
    for (int y = 0; y < m->rows; y++) {
        for (int x = 0; x < m->columns; x++) {
            *(f++) -= a->data[x] * b->data[y];
        }
    }
}

/*------------------------------------------*/

/*      NEURAL NETWORK MODEL OPERATIONS     */

/*------------------------------------------*/

void model_init(const Model* restrict model)
{
    Layer* layer = model->layers;
    for (int i = 0; i < model->layer_count - 1; i++) {
        int size = layer->w.columns * layer->w.rows;
        float* f = layer->w.data;
        
        for (int j = 0; j < size; j++) {
            *(f++) = rand_gauss() / (float)layer->a.size;
        }
        layer++;
    }
}

void model_forward(const Model* restrict model)
{
    Layer* layer = model->layers, *next_layer; next_layer = layer + 1;
    for (int i = 0; i < model->layer_count - 1; i++) {
        
        vector_swap(vector_by_matrix_plus(&layer->w, &layer->a, &next_layer->b), &next_layer->z);

        if (i < model->layer_count - 2) vector_swap(vector_sigmoid(&next_layer->z), &next_layer->a);
        else vector_swap(vector_sigmoid(&next_layer->z), &next_layer->a);

        next_layer++;
        layer++;
    }
}

void model_backwards(const Model* restrict model, const Vec* restrict desired_output)
{
    Layer* layer = model->layers + model->layer_count - 1;
    
    Vec c = vector_copy(&layer->a);
    vector_sub(&c, desired_output);
    vector_scale(&c, 2.0f);

    vector_swap(vector_dsigmoid(&layer->z), &layer->d);
    vector_hadamard(&layer->d, &c);

    vector_free(&c);
    
    Layer* next_layer = layer--;
    for (Layer* end = model->layers - 1; layer != end; layer--) {
        
        c = vector_by_matrix_transposed(&layer->w, &next_layer->d);
        vector_swap(vector_dsigmoid(&layer->z), &layer->d);

        vector_hadamard(&layer->d, &c);
        vector_free(&c);

        next_layer--;
    }
}

float model_cost(const Model* restrict model, const Vec* restrict desired_output)
{
    Layer* layer = model->layers + model->layer_count - 1;
    Vec* a = &layer->a;
    
    float cost = 0.0f;
    for (int i = 0; i < a->size; i++) {
        float f = a->data[i] - desired_output->data[i];
        cost += f * f;
    }

    return cost;
}

void model_update(const Model* restrict model, float alpha)
{
    Layer* layer = model->layers, *next_layer; next_layer = layer + 1;
    for (Layer* end = layer + model->layer_count - 1; layer != end; layer++) {
        
        vector_scale(&next_layer->d, alpha);
        vector_sub(&next_layer->b, &next_layer->d);
        matrix_minus_vec_by_vec(&layer->w, &layer->a, &next_layer->d);
        
        next_layer++;
    }
}