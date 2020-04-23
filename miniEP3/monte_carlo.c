#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef VERBOSE
#define VERBOSE 0
#endif

#define FUNCTIONS 1

struct timer_info {
  clock_t c_start;
  clock_t c_end;
  struct timespec t_start;
  struct timespec t_end;
  struct timeval v_start;
  struct timeval v_end;
};

struct timer_info timer;

char *usage_message = "usage: ./monte_carlo SAMPLES FUNCTION_ID N_THREADS\n";

struct function {
  long double (*f)(long double);
  long double interval[2];
};

long double rand_interval[] = {0.0, (long double)RAND_MAX};

long double f1(long double x) { return 2 / (sqrt(1 - (x * x))); }

struct function functions[] = {{&f1, {0.0, 1.0}}};

// Your thread data structures go here

typedef struct {
  int thread_id;
  long int interval_start;
  long int interval_end;
  long double (*f)(long double);
  long double *samples;
  long double *results;
} thread_data;

thread_data *thread_data_array;

// End of data structures

long double *samples;
long double *results;

long double map_intervals(long double x, long double *interval_from,
                          long double *interval_to) {
  x -= interval_from[0];
  x /= (interval_from[1] - interval_from[0]);
  x *= (interval_to[1] - interval_to[0]);
  x += interval_to[0];
  return x;
}

long double *uniform_sample(long double *interval, long double *samples,
                            int size) {
  for (int i = 0; i < size; i++) {
    samples[i] = map_intervals((long double)rand(), rand_interval, interval);
  }
  return samples;
}

void print_array(long double *sample, int size) {
  printf("array of size [%d]: [", size);

  for (int i = 0; i < size; i++) {
    printf("%Lf", sample[i]);

    if (i != size - 1) {
      printf(", ");
    }
  }

  printf("]\n");
}

long double monte_carlo_integrate(long double (*f)(long double),
                                  long double *samples, int size) {
  long double accumulator = 0.0L;

  for (int i = 0; i < size; i++) {
    accumulator += f(samples[i]);
  }

  return accumulator / size;
}

void *monte_carlo_integrate_thread(void *args) {
  thread_data *td = (thread_data *)args;

  int thread_id = td->thread_id;
  long int left = td->interval_start;
  long int right = td->interval_end;

  long int num_iterations = right - left;
  long double (*f)(long double) = td->f;
  long double *samples = td->samples;
  long double *results = td->results;
  long double accumulator = 0.0L;

  for (int counter = left; counter < right; counter++) {
    accumulator += f(samples[counter]);
  }

  results[thread_id] = accumulator / num_iterations;

  pthread_exit(NULL);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("%s", usage_message);
    exit(-1);
  } else if (atoi(argv[2]) >= FUNCTIONS || atoi(argv[2]) < 0) {
    printf("Error: FUNCTION_ID must in [0,%d]\n", FUNCTIONS - 1);
    printf("%s", usage_message);
    exit(-1);
  } else if (atoi(argv[3]) < 0) {
    printf("Error: I need at least 1 thread\n");
    printf("%s", usage_message);
    exit(-1);
  }

  if (DEBUG) {
    printf("Running on: [debug mode]\n");
    printf("Samples: [%s]\n", argv[1]);
    printf("Function id: [%s]\n", argv[2]);
    printf("Threads: [%s]\n", argv[3]);
    printf("Array size on memory: [%.2LFGB]\n",
           ((long double)atoi(argv[1]) * sizeof(long double)) / 1000000000.0);
  }

  srand(time(NULL));

  int size = atoi(argv[1]);
  struct function target_function = functions[atoi(argv[2])];
  int n_threads = atoi(argv[3]);
  pthread_t threads[n_threads];

  samples = malloc(size * sizeof(long double));
  thread_data_array = malloc(n_threads * sizeof(thread_data));
  results = malloc(n_threads * sizeof(long double));

  // Calculating the samples only once so that each thread can process 1/n of
  // the total size
  long double *static_samples =
      uniform_sample(target_function.interval, samples, size);

  long double estimate;

  if (n_threads == 1) {
    if (DEBUG) {
      printf("Running sequential version\n");
    }

    timer.c_start = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
    gettimeofday(&timer.v_start, NULL);

    estimate = monte_carlo_integrate(target_function.f, static_samples, size);

    timer.c_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
    gettimeofday(&timer.v_end, NULL);
  } else {
    if (DEBUG) {
      printf("Running parallel version\n");
    }

    long double acc = 0.0;
    float work_per_thread = (size / n_threads);
    thread_data base_data;

    base_data.f = target_function.f;
    base_data.results = results;
    base_data.samples = static_samples;

    timer.c_start = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
    gettimeofday(&timer.v_start, NULL);

    for (int i = 0; i < n_threads; i++) {
      long int left_boundary = i * (int)work_per_thread;
      long int right_boundary = left_boundary + (int)work_per_thread;

      // The last thread will perform a bit more work
      if (i == n_threads - 1) {
        right_boundary += (size % n_threads);
      }

      thread_data_array[i] = base_data;
      thread_data_array[i].thread_id = i;
      thread_data_array[i].interval_start = left_boundary;
      thread_data_array[i].interval_end = right_boundary;

      pthread_create(&threads[i], NULL, monte_carlo_integrate_thread,
                     &thread_data_array[i]);
    }

    for (int i = 0; i < n_threads; i++) {
      pthread_join(threads[i], NULL);
      acc += results[i];
    }

    estimate = acc / n_threads;

    timer.c_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
    gettimeofday(&timer.v_end, NULL);

    if (DEBUG && VERBOSE) {
      print_array(results, n_threads);
    }
  }

  if (DEBUG) {
    if (VERBOSE) {
      print_array(samples, size);
      printf("Estimate: [%.33LF]\n", estimate);
    }
    printf(
        "%.16LF, [%f, clock], [%f, clock_gettime], [%f, gettimeofday]\n",
        estimate,
        (double)(timer.c_end - timer.c_start) / (double)CLOCKS_PER_SEC,
        (double)(timer.t_end.tv_sec - timer.t_start.tv_sec) +
            (double)(timer.t_end.tv_nsec - timer.t_start.tv_nsec) /
                1000000000.0,
        (double)(timer.v_end.tv_sec - timer.v_start.tv_sec) +
            (double)(timer.v_end.tv_usec - timer.v_start.tv_usec) / 1000000.0);
  } else {
    printf("%.16LF, %f\n", estimate,
           (double)(timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double)(timer.t_end.tv_nsec - timer.t_start.tv_nsec) /
                   1000000000.0);
  }
  return 0;
}
