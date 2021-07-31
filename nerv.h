#ifndef NERV_H
#define NERV_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************
 * nerv tiny C tensor framework
 * ----------------------------
 ****************** @eulogic */

typedef struct Vec {
    int size;
    float* data;
} Vec;

typedef struct Mat {
    int rows, columns;
    float* data;
} Mat;

typedef struct {
    Mat w;
    Vec b, z, a, d;
} Layer;

typedef struct {
    int layer_count;
    Layer* layers;
} Model;

/*------------------------------------------*/

/*    PSEUDO RANDOM GENERATOR AND MATH      */

/*********************************************
 * pseudo random number distribution generator 
 * ******************************************/

void rand_seed(unsigned int seed);
unsigned int rand_uint();
double rand_gauss();
double rand_norm();
double rand_dist(double standard_deviation, double mean);

/*********************************************
 *   floating point functions and operations
 * ******************************************/

unsigned int ftou(float n);
float utof(unsigned int u);
float clampf(float x, float min, float max);
float normf(float x);
float sigmoid(float x);
float sigderiv(float sig);
float dsigmoid(float x);
float relu(float x);
float drelu(float x);
float leaky_relu(float x, float slope);
float dleaky_relu(float x, float slope);

/*------------------------------------------*/

/*  VECTOR DATA STRUCTURE AND OPERATIONS    */

/*********************************************
 *      vector creation and management
 * ******************************************/

Vec vector(int size);
Vec vector_create(int size, ...);
Vec vector_uniform(int size, float val);
Vec vector_copy(Vec* src);
void vector_free(Vec* vector);

/*********************************************
 *     vector functions and operations
 * ******************************************/

void vector_scale(Vec* v, float n);
void vector_add(Vec* dst, Vec* src);
void vector_sub(Vec* dst, Vec* src);
void vector_hadamard(Vec* dst, Vec* src);

Vec vector_by_matrix(Mat* mat, Vec* vec);
Vec vector_by_matrix_transposed(Mat* mat, Vec* vec);
Vec vector_sigmoid(Vec* v);
Vec vector_dsigmoid(Vec* v);
Vec vector_sigderiv(Vec* v);
Vec vector_relu(Vec* v);
Vec vector_drelu(Vec* v);
Vec vector_leaky_relu(Vec* v, float leak);
Vec vector_dleaky_relu(Vec* v, float leak);
Vec vector_softmax(Vec* v);

/*------------------------------------------*/

/*   MATRIX DATA STRUCTURE AND OPERATIONS   */

/*********************************************
 *       matrix creation and management
 * ******************************************/

Mat matrix(int rows, int columns);
Mat matrix_create(int rows, int columns, ...);
Mat matrix_identity(int size);
Mat matrix_uniform(int rows, int columns, float val);
Mat matrix_copy(Mat* mat);
Mat matrix_vector(Vec* v);
void matrix_free(Mat* mat);

/*********************************************
 *       matrix functions and operations
 * ******************************************/

Mat matrix_scale(Mat* mat, float scale);
Mat matrix_multiply(Mat* a, Mat* b);
Mat matrix_hadamard(Mat* a, Mat* b);
Mat matrix_transpose(Mat* m);

/*------------------------------------------*/

/*   NEURAL NETWORK AND LAYER STRUCTURES    */

//      z = w * a + b
//      a = f(z)
//      c = (a - y)^2
//      d = 2(a - y) * f'(z)
//      dw = a * d * g
//      db = d * g

/*********************************************
 *          fully connected layers 
 * ******************************************/

Layer layer_create(int layer_size, int next_layer_size);
Layer layer_copy(Layer* layer);
void layer_vector_free(Layer* layer);
void layer_matrix_free(Layer* layer);
void layer_free(Layer* layer);

/*********************************************
 *    neural network model data structure 
 * ******************************************/

Model model_new(int layer_count);
Model model_create(int layer_count, ...);
Model model_copy(Model* model);
void model_free(Model* model);

/*********************************************
 *    neural network model operations 
 * ******************************************/

void model_init(Model* model);
void model_forward(Model* model);
void model_backwards(Model* model, Vec* desired_output);
void model_update(Model* model, float alpha);
float model_cost(Model* model, Vec* desired_output);

/*------------------------------------------*/

/*                 NERV IO                  */

/*********************************************
 *     serialize models - save and load
 * ******************************************/

Model model_load(char* path);
void model_save(char* path, Model* model);

/*********************************************
 *   useful IO functions to print and scan
 * ******************************************/

Vec vector_scan();
void vector_print(Vec* vec);

Mat matrix_scan();
void matrix_print(Mat* mat);

Model model_scan();
void model_print(Model* nm);
void model_print_input(Model* nm);
void model_print_output(Model* nm);
void model_print_struct(Model* nm);

/*------------------------------------------*/

#ifdef __cplusplus
}
#endif
#endif
