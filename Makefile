.PHONY: all clean

libcrossvalidation.so: cross_validation.o
	$(CC) -fPIC -shared -O3 cross_validation.o -o libcrossvalidation.so `pkg-config --libs gsl`

cross_validation.o: cross_validation.c cross_validation.h
	$(CC) -c -fPIC -O3 cross_validation.c `pkg-config --cflags gsl` -o cross_validation.o

demo: libcrossvalidation.so
	$(CC) demo.c ./libcrossvalidation.so `pkg-config --cflags --libs gsl` -o demo

clean:
	rm -f *.o *.so demo
