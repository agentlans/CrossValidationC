#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_randist.h>

#include "cross_validation.h"

void copy_ints(int* dest, int* src, int n) {
    if (n == 0) return;
    memmove(dest, src, n*sizeof(int));
}

void cross_validation_free(cross_validation* cv) {
    if (cv) free(cv->indices);
    free(cv);
}

// Returns an independent deep copy of the cross validation object
cross_validation* cross_validation_copy(cross_validation* cv) {
    // Reserve memory for copy
    cross_validation* copy = malloc(sizeof(cross_validation));
    if (!copy) return NULL;
    // Indices of the copy
    int* ind_copy = calloc(cv->n, sizeof(int));
    if (!ind_copy) {
        free(copy);
        return NULL;
    }
    copy_ints(ind_copy, cv->indices, cv->n);
    // Set fields of the copy
    copy->indices = ind_copy;
    copy->n = cv->n;
    copy->k = cv->k;
    copy->floor_nk = cv->floor_nk;
    copy->n_mod_k = cv->n_mod_k;
    return copy;
}

void cross_validation_shuffle(cross_validation* cv, gsl_rng* r) {
    gsl_ran_shuffle(r, cv->indices, cv->n, sizeof(int));
}

cross_validation* cross_validation_new(int n, int folds, gsl_rng* r) {
    cross_validation* cv = malloc(sizeof(cross_validation));
    if (!cv) return NULL;
    // Indices in [0,n)
    cv->indices = calloc(n, sizeof(int));
    for (int i = 0; i < n; ++i) {
        cv->indices[i] = i;
    }
    if (!cv->indices) {
        free(cv);
        return NULL;
    }
    // Save the number of folds and precomputed values
    cv->n = n;
    cv->k = folds;
    cv->floor_nk = floor(n / folds);
    cv->n_mod_k = n % folds;
    // Shuffle order of indices
    cross_validation_shuffle(cv, r);
    return cv;
}

static inline int min_int(int a, int b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

int cross_validation_fold_index(cross_validation* cv, int i) {
    return i * cv->floor_nk + min_int(i, cv->n_mod_k);
}

void cross_validation_train_set(int* train, int* train_size, cross_validation* cv, int i) {
    int ex_begin = cross_validation_fold_index(cv, i);
    int ex_end = cross_validation_fold_index(cv, i+1);
    int ex_size = ex_end - ex_begin;
    // Copy the part before and after the testing set
    copy_ints(train, cv->indices, ex_begin);
    copy_ints(train+ex_begin, cv->indices+ex_end, cv->n-ex_end);
    *train_size = cv->n - ex_size;
}

void cross_validation_test_set(int* test, int* test_size, cross_validation* cv, int i) {
    int ex_begin = cross_validation_fold_index(cv, i);
    int ex_end = cross_validation_fold_index(cv, i+1);
    int ex_size = ex_end - ex_begin;
    // Copy the testing set part
    copy_ints(test, cv->indices+ex_begin, ex_size);
    *test_size = ex_size;
}

void cross_validation_train_test_set(int* train, int* train_size, int* test, int* test_size, 
    cross_validation* cv, int i) {

    int ex_begin = cross_validation_fold_index(cv, i);
    int ex_end = cross_validation_fold_index(cv, i+1);
    int ex_size = ex_end - ex_begin;

    // Copy the part before and after the testing set
    copy_ints(train, cv->indices, ex_begin);
    copy_ints(train+ex_begin, cv->indices+ex_end, cv->n-ex_end);
    *train_size = cv->n - ex_size;

    // Copy the testing set part
    copy_ints(test, cv->indices+ex_begin, ex_size);
    *test_size = ex_size;
}
