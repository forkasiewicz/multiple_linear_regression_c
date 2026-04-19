all:
	gcc -O3 -fopenmp main.c -o main -lm && ./main
