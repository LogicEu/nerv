
/*********************************************
 *   useful IO functions to print and scan
 * ******************************************/

#include <nerv.h>
#include <stdio.h>
#include <string.h>

Vec vector_scan()
{
    int size;
    printf("Enter size of vector: ");
    scanf("%d", &size);
    
    Vec v = vector(size);
    for (int i = 0; i < size; i++) {
        printf("%d: ", i + 1);
        scanf("%f", v.data + i);
    }

    vector_print(&v);
    return v;
}

void vector_print(const Vec* restrict vec)
{
    printf("Vector\nSize: %d\n", vec->size);
    float* f = vec->data;
    for (float* end = f + vec->size; f != end; f++) {
        printf("[ %f ]\n", *f);
    }
}

Mat matrix_scan()
{
    int rows, columns;
    printf("Enter rows: ");
    scanf("%d", &rows);
    printf("Enter columns: ");
    scanf("%d", &columns);

    Mat mat = matrix(rows, columns);
    float* f = mat.data;
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            printf("%dx%d: ", y + 1, x + 1);
            scanf("%f", (f++));
        }
    }

    matrix_print(&mat);
    return mat;
}

void matrix_print(const Mat* restrict mat)
{
    printf("Matrix\nRows: %u\tColumns: %u\n", mat->rows, mat->columns);

    float* f = mat->data;
    for (int y = 0; y < mat->rows; y++) {
        printf("[");
        for (int x = 0; x < mat->columns; x++) {
            printf(" %f ", *(f++));
        }
        printf("]\n");
    }
    printf("\n");
}

Model model_scan()
{
    int layer_count;
    printf("Enter number of layers: ");
    scanf("%d", &layer_count);

    Model model = model_new(layer_count);

    int layer_sizes[layer_count + 1];
    memset(&layer_sizes[0], 0, sizeof(int) * (layer_count + 1));
    
    for (int i = 0; i < layer_count; i++) {
        printf("Enter size of layer %d: ", i + 1);
        scanf("%d", &layer_sizes[i]);
    }

    Layer* layer = model.layers;
    for (int i = 0; i < layer_count; i++) {
        int size = layer_sizes[i], next = layer_sizes[i + 1];
        *(layer++) = layer_create(size, next);
    }

    model_print_struct(&model);
    return model;
}

void model_print(const Model* restrict model)
{
    printf("Model:\n");
    Layer* layer = model->layers;
    for (Layer* end = layer + model->layer_count; layer != end; layer++) {
        vector_print(&layer->a);
    }
}

void model_print_input(const Model* restrict model)
{
    printf("Model Input:\n");
    vector_print(&model->layers->a);
}

void model_print_output(const Model* restrict model)
{
    printf("Model Output:\n");
    vector_print(&model->layers[model->layer_count - 1].a);
}

void model_print_struct(const Model* restrict model)
{
    printf("Model Structure\nLayers: %d\n", model->layer_count);
    int i = 0;
    
    Layer* layer = model->layers;
    for (Layer* end = layer + model->layer_count; layer != end; layer++) {
        printf("Layer %d - Params: %d\n", ++i, layer->a.size);
    }
}