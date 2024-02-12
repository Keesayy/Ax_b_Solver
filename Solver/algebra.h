#ifndef _LINEAR_ALGEBRA_H_
#define _LINEAR_ALGEBRA_H_

#include <stdint.h>
#include <stddef.h>

#define ABS(A) ( ((A) > 0) ? (A) : (-(A)) )
#define SWAP(TYPE, A, B) do {TYPE tmp = (A); (A) = (B); (B) = tmp;} while(0)
#define M(A, I, J) (A).elts[(I)*(A).cols + (J)]
#define MATFREE(M) free((M.elts))

// Precision float or double
typedef double vector;

/** Matrix type **/
typedef struct{
	size_t rows;
	size_t cols;
	vector *elts;
}Mat;

/** **/
vector rand_vector(void);
Mat Mat_alloc(size_t rows, size_t cols);
void Mat_print(Mat m);

/** **/
void Mat_rand(Mat m, vector low, vector high);
void Mat_set(vector k, Mat m);
Mat Array_as_Mat(vector *array, size_t rows, size_t cols);
Mat Mat_copy(Mat m);
Mat Mat_transpose(Mat m);
Mat Mat_col(Mat m, size_t j);
Mat Mat_row(Mat m, size_t i);
Mat Mat_cat(Mat a, Mat b);
Mat Mat_cat_list(Mat *a, size_t n);
void Swap_rows(Mat m, size_t i, size_t j);

/** Operations **/
void Mat_sum(Mat dst, Mat a);
Mat Mat_product(Mat a, Mat b);
void Mat_Kproduct(double k, Mat a);
vector Mat_dot(Mat x, Mat y);
vector Sarrus(Mat m);

/***************** Solve Ax = b *****************/

typedef enum{LOWER, UPPER} TMat;

/** A square and non singular **/
Mat Triangle_solve(Mat ab, TMat type); 
size_t Gaussian_elim(Mat ab);
uint *LU_factor(Mat a);
Mat LU_solve(Mat LU, uint *sigma, Mat b);

/** Ax = b inconsistent (A rectangular) or A singular ? => least squares method **/
typedef struct{
	Mat Q; // Orthogonal
	Mat R; // UpperT
}QR;

Mat HH_vector(Mat x);
QR HH_factor(Mat a); // Not working
QR GS_factor(Mat a);

/** For whatever A **/
Mat Solve(Mat a, Mat b);

#endif
