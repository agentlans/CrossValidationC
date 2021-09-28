#include <stdio.h>
#include "cross_validation.h"

void print_ints(const char* heading, int* arr, int n) {
    printf("%s: ", heading);
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Start a random number generator
    gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937);

    if (argc != 3) {
        printf("Generates example indices for k-fold cross validation.\n");
        printf("Usage: ./demo [dataset size] [number of folds]\n");
        return 1;
    }
    //int n = 20;
    //int folds = 7;
    int n = atoi(argv[1]);
    int folds = atoi(argv[2]);
    if (n <= 0) {
        printf("Invalid dataset size.\n"); return 1;
    }
    if (folds <= 0 || folds > n) {
        printf("Invalid number of folds.\n"); return 1;
    }

    // Set up cross validation object
    cross_validation* cv = cross_validation_new(n, folds, r);
    //cross_validation_shuffle(cv, r);

    // Must first allocate space to hold indices for train and test sets
    int train[n - n/folds + 1];
    int test[n/folds + 1];
    int test_size = 0;
    int train_size = 0;

    for (int fold = 0; fold < folds; ++fold) {
        // Get the indices for the train and test sets for this particular fold
        cross_validation_train_test_set(train, &train_size, test, &test_size, cv, fold);
        // Print the indices
        printf("Fold %d\n", fold);
        print_ints("Training set", train, train_size);
        print_ints("Testing set", test, test_size);
        printf("\n");
    }
    // Clean up
    cross_validation_free(cv);
    gsl_rng_free(r);
    return 0;
}
