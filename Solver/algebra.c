#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "lalgebra.h"

Mat Mat_alloc(size_t rows, size_t cols){
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.elts = (vector*)malloc(rows * cols * sizeof(*m.elts));
	assert(m.elts != NULL);

	return m;
}

void Mat_print(Mat m){
	for(size_t i=0; i<m.rows; ++i){
		for(size_t j=0; j<m.cols; ++j)
			printf("%0.2f ", M(m, i, j));
		printf("\n");
	}
	printf("\n");
}

vector rand_float(void){return (vector) rand() / (vector) RAND_MAX;}

void Mat_rand(Mat m, vector low, vector high){
	for(size_t i=0; i<m.rows; ++i)
		for(size_t j=0; j<m.cols; ++j)
			M(m, i, j) = rand_float()*(high - low) + low; 
}

void Mat_set(vector k, Mat m){
	for(size_t i=0; i<m.rows; ++i)
		for(size_t j=0; j<m.cols; ++j)
			M(m, i, j) = k; 
}

Mat Array_as_Mat(vector *array, size_t rows, size_t cols){
	return (Mat){.rows = rows, . cols = cols, .elts = array};
}

void Mat_sum(Mat dst, Mat a){
	assert(dst.rows == a.rows);
	assert(dst.cols == a.cols);

	for(size_t i=0; i<dst.rows; ++i)
		for(size_t j=0; j<dst.cols; ++j)
			M(dst, i, j) += M(a, i, j); 
}

Mat Mat_product(Mat a, Mat b){
	assert(a.cols == b.rows);
	Mat dst = Mat_alloc(a.rows, b.cols);

	for(size_t i=0; i<dst.rows; ++i){
		for(size_t j=0; j<dst.cols; ++j){
			M(dst, i, j) = 0;
			for(size_t k=0; k<a.cols; ++k)
				M(dst, i, j) += M(a, i, k) * M(b, k, j);
		}
	}
	return dst;
}

void Mat_Kproduct(double k, Mat a){
	for(size_t i=0; i<a.rows; ++i)
		for(size_t j=0; j<a.cols; ++j)
			M(a, i, j) *= k;
}

vector Mat_dot(Mat x, Mat y){
	assert((x.cols == y.cols) == 1);
	assert(x.rows == y.rows);

	Mat xt = Mat_transpose(x);
	Mat xty = Mat_product(xt, y);

	vector tmp = M(xty, 0, 0);
	MATFREE(xty);
	MATFREE(xt);
	return tmp;
}

vector Sarrus(Mat m){
	double det = 0.0;
  det =   M(m, 0, 0) * M(m, 1, 1) * M(m, 2, 2)
        + M(m, 0, 1) * M(m, 1, 2) * M(m, 2, 0)
        + M(m, 0, 2) * M(m, 1, 0) * M(m, 2, 1)
        - M(m, 0, 2) * M(m, 1, 1) * M(m, 2, 0)
        - M(m, 0, 1) * M(m, 1, 0) * M(m, 2, 2)
        - M(m, 0, 0) * M(m, 1, 2) * M(m, 2, 1);

  return det;
}

Mat Mat_copy(Mat m){
	Mat c = Mat_alloc(m.rows, m.cols);

	for(size_t i=0; i<c.rows; ++i)
		for(size_t j=0; j<c.cols; ++j)
			M(c, i, j) = M(m, i, j);

	return c;
}

Mat Mat_transpose(Mat m){
	Mat c = Mat_copy(m);
	SWAP(size_t, c.rows, c.cols);

	if (!(c.rows == 1 || c.cols == 1)){
		for(size_t i=0; i<c.rows; ++i)
			for(size_t j=0; j<c.cols; ++j)
				M(c, i, j) = M(m, j, i);
	}
	return c;
}

Mat Mat_col(Mat m, size_t j){
	Mat a = Mat_alloc(m.rows, 1);
	for(size_t i=0; i<m.rows; ++i)
		M(a, i, 0) = M(m, i, j);
	return a;
}

Mat Mat_row(Mat m, size_t i){
	Mat a = Mat_alloc(1, m.cols);
	for(size_t j=0; j<m.cols; ++j)
		M(a, 0, j) = M(m, i, j);
	return a;
}

Mat Mat_cat(Mat a, Mat b){
	assert(a.rows == b.rows);
	Mat ab = Mat_alloc(a.rows, a.cols + b.cols);

	for(size_t i=0; i<ab.rows; ++i){
		for(size_t j=0; j<a.cols; ++j)
			M(ab, i, j) = M(a, i, j);

		for(size_t j=0; j<b.cols ; ++j)
			M(ab, i, j+a.cols) = M(b, i, j);
	}	
	return ab;
}

Mat Mat_cat_list(Mat *a, size_t n){
	assert(n >= 2);
	Mat q = Mat_cat(a[0], a[1]);

	for(size_t k=2; k<n; ++k){
		Mat tmp = q;
		q = Mat_cat(tmp, a[k]);
		MATFREE(tmp);
	}
	return q;
}

/***************** Solving Ax = b *****************/

//A non singular
Mat Triangle_solve(Mat ab, TMat type){
	assert(ab.rows == ab.cols - 1);
	const size_t n = ab.rows - 1;

	Mat x = Mat_alloc(ab.rows, 1);
	switch(type){
	case UPPER:
		M(x, n, 0) = M(ab, n, n+1) / M(ab, n, n);

		for(int k=n-1; k>=0; --k){
			vector sum = 0;
			for(size_t j=k+1; j<=n; ++j)
				sum += M(ab, k, j) * M(x, j, 0);
			M(x, k, 0) = (M(ab, k, n+1) - sum) / M(ab, k, k);
		}
		break;

	case LOWER:
		M(x, 0, 0) = M(ab, 0, n+1) / M(ab, 0, 0);

		for(size_t k=1; k<=n; ++k){
			vector sum = 0;
			for(size_t j=0; j<=k-1; ++j)
				sum += M(ab, k, j) * M(x, j, 0);
			M(x, k, 0) = (M(ab, k, n+1) - sum) / M(ab, k, k);
		}

	default:
		break;	
	}
	
	return x;
}

void Swap_rows(Mat m, size_t i, size_t j){
	for(size_t k=0; k<m.cols; ++k){
		vector tmp = M(m, i, k);
		M(m, i, k) = M(m, j, k);
		M(m, j, k) = tmp;
	}
}

size_t Gaussian_elim(Mat ab){
	assert(ab.rows == ab.cols - 1);
	const size_t n = ab.rows - 1;
	vector cp = 0;
	size_t ip, rank = 0;
	
	for(size_t k=0; k<n; ++k){
		//Search Pivot 
		cp = ABS(M(ab, k, k));
		ip = k;
		for(size_t i=k+1; i<=n; ++i){
			if(ABS(M(ab, i, k) > cp)){
				cp = ABS(M(ab, i, k));
				ip = i;
			}
		}
		//Permute
		if(ip != k) Swap_rows(ab, ip, k);

		//Pivoting
		for(size_t i=k+1; i<=n; ++i){
			for(size_t j=k+1; j<=n+1; ++j)
				M(ab, i, j) -= (M(ab, i, k) / M(ab, k, k)) * M(ab, k, j);
			M(ab, i, k) = 0;
		}
	}
	for(size_t k=0; k<ab.rows; ++k) if(M(ab, k, k) != 0) rank++;
	
	return rank;
}

uint *LU_factor(Mat a){
	assert(a.rows == a.cols);
	uint *sigma = (uint*)malloc(a.rows*sizeof(uint));
	for(size_t i=0; i<a.rows; ++i) sigma[i] = i;
	
	const size_t n = a.rows - 1;
	vector cp = 0;
	size_t ip = 0;
	
	for(size_t k=0; k<n; ++k){
		//Search Pivot 
		cp = ABS(M(a, k, k));
		ip = k;
		for(size_t i=k+1; i<=n; ++i){
			if(ABS(M(a, i, k) > cp)){
				cp = ABS(M(a, i, k));
				ip = i;
			}
		}
		//Permute
		if(ip != k){
			SWAP(uint, sigma[ip], sigma[k]);
			Swap_rows(a, ip, k);
		}
		//Pivoting
		for(size_t i=k+1; i<=n; ++i){
			M(a, i, k) /= M(a, k, k);
			for(size_t j=k+1; j<=n; ++j)
				M(a, i, j) -= M(a, i, k) * M(a, k, j);
		}
	}
	return sigma;
}

Mat LU_solve(Mat LU, uint *sigma, Mat b){
	assert(LU.rows == LU.cols);
	assert(LU.rows == b.rows);
	Mat x = Mat_alloc(LU.rows, 1);
	const size_t n = LU.rows - 1;

	//Ly = b
	M(x, 0, 0) = M(b, sigma[0], 0);
	for(size_t k=1; k<=n; ++k){
		vector sum = 0;
		for(size_t j=0; j<=k-1; ++j)
			sum += M(LU, k, j) * M(x, j, 0);
		M(x, k, 0) = M(b, sigma[k], 0) - sum;
	}
	//Ux = y
	M(x, n, 0) /= M(LU, n, n);
	for(int k=n-1; k>=0; --k){
		vector sum = 0;
		for(size_t j=k+1; j<=n; ++j)
			sum += M(LU, k, j) * M(x, j, 0);
		M(x, k, 0) = (M(x, k, 0) - sum) / M(LU, k, k);
	}
	return x;
}

Mat HH_vector(Mat x){
	assert(x.cols == 1);

	Mat xt = Mat_transpose(x);
	Mat nx = Mat_product(xt, x);
	assert((nx.cols == nx.rows) == 1);

	vector alpha = M(x, 0, 0) / ABS(M(x, 0, 0));
	vector k = 1.0 / (sqrt(M(nx, 0, 0)) * (1 + alpha));

	Mat v = Mat_alloc(x.rows, 1);
	Mat_Kproduct(k , v);
	M(v, 0, 0) = 1.0;
	
	MATFREE(xt);
	MATFREE(nx);
	return v;
}

// Not working
QR HH_factor(Mat a){
	QR b;
	b.Q = Mat_copy(a);
	b.R = Mat_alloc(a.cols, a.cols);
	Mat_set(0.0, b.R);

	Mat v, vt, vvt;
	Mat x = Mat_alloc(a.rows, 1);
	for(size_t k=0; k<a.cols; ++k){
		// x is subvector of the k-th colum below the diagonal
		Mat_set(0.0, x);
		for(size_t i=k; i<a.rows; ++i)
			M(x, i, 0) = M(b.Q, i, k);

		v = HH_vector(x);
		vt = Mat_transpose(v);
		vvt = Mat_product(v, vt);

		//HH transformation on columns of Q and R
		Mat tmp1 = Mat_product(vvt, b.Q);
		Mat_Kproduct(2.0, tmp1);
	
	}
	MATFREE(v);
	MATFREE(vt);
	MATFREE(vvt);
	MATFREE(x);
	return b;
}

QR GS_factor(Mat a){
	QR b;
	b.R = Mat_alloc(a.cols, a.cols);
	Mat *q = (Mat*)malloc(a.cols * sizeof(Mat));

	q[0] = Mat_col(a, 0);
	M(b.R, 0, 0) = sqrt(Mat_dot(q[0], q[0]));
	Mat_Kproduct(1.0 / M(b.R, 0, 0), q[0]);


	Mat tmp;
	for(size_t j=1; j<a.cols; ++j){
		q[j] = Mat_col(a, j);

		for(size_t i=0; i<=j-1; ++i){
			M(b.R, i, j) = Mat_dot(q[j], q[i]);

			tmp = Mat_copy(q[i]);
			Mat_Kproduct((-M(b.R, i, j)), tmp);
			Mat_sum(q[j], tmp);
			MATFREE(tmp);
		}
		M(b.R, j, j) = sqrt(Mat_dot(q[j], q[j]));
		Mat_Kproduct(1.0 / M(b.R, j, j), q[j]);
	}
	b.Q = Mat_cat_list(q, a.cols);

	for(size_t k=0; k<a.cols; ++k) MATFREE(q[k]);
	free(q);
	return b;
}

Mat Solve(Mat a, Mat b){
	assert(a.rows == b.rows);
	assert(b.cols == 1);
	Mat x;

	if(a.rows == a.cols){
		Mat ab = Mat_cat(a, b);
		if(Gaussian_elim(ab) == a.rows){
			x = Triangle_solve(ab, UPPER);
			MATFREE(ab);
			return x;
		}
		MATFREE(ab);
	}
	//a.rows != a.cols or A singular 
	QR qr = GS_factor(a);
	Mat qt = Mat_transpose(qr.Q);
	Mat qtb = Mat_product(qt, b);
	Mat ab = Mat_cat(qr.R, qtb);
	x = Triangle_solve(ab, UPPER);

	MATFREE(qr.Q);
	MATFREE(qr.R);
	MATFREE(qt);
	MATFREE(qtb);
	MATFREE(ab);
	return x;
}

int main(void){
	size_t n = 10;
	srand(time(NULL));

	Mat a = Mat_alloc(n, n/2);
	Mat_rand(a, 0, 1);
	Mat_print(a);

	Mat b = Mat_alloc(n, 1);
	Mat_rand(b, 0, 1);
	Mat_print(b);

	QR g = GS_factor(a);
	Mat_print(g.Q);
	Mat_print(g.R);

	Mat qr = Mat_product(g.Q, g.R);
	Mat_print(qr);

	Mat x = Solve(a, b);
	Mat_print(x);

	Mat c = Mat_product(a, x);
	Mat_print(c);

	MATFREE(x);
	MATFREE(c);
	MATFREE(a);
	MATFREE(b);
	MATFREE(qr);
	MATFREE(g.Q);
	MATFREE(g.R);
	return 0;
}
