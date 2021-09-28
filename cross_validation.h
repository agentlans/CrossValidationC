#ifndef _CROSSVALIDATION
#define _CROSSVALIDATION

#include <gsl/gsl_rng.h>

typedef struct cross_validation {
    int* indices;
    int n;
    int k; // Number of folds
    // Precomputed values
    int floor_nk;
    int n_mod_k;
} cross_validation;


cross_validation* cross_validation_new(int n, int folds, gsl_rng* r);
void cross_validation_free(cross_validation* cv);
// Returns an independent deep copy of the cross validation object
cross_validation* cross_validation_copy(cross_validation* cv);

void cross_validation_shuffle(cross_validation* cv, gsl_rng* r);

// To obtain the indices of each cross-validation fold
void cross_validation_train_set(int* train, int* train_size, cross_validation* cv, int i);
void cross_validation_test_set(int* test, int* test_size, cross_validation* cv, int i);
void cross_validation_train_test_set(int* train, int* train_size, int* test, int* test_size, 
    cross_validation* cv, int i);

#endif