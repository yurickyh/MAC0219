#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
unsigned char **image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
int colors[17][3] = {
    {66, 30, 15},    {25, 7, 26},     {9, 1, 47},      {4, 4, 73},
    {0, 7, 100},     {12, 44, 138},   {24, 82, 177},   {57, 125, 209},
    {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95},
    {255, 170, 0},   {204, 128, 0},   {153, 87, 0},    {106, 52, 3},
    {16, 16, 16},
};

// THREAD VARIABLES

int num_threads;

typedef struct {
    int	thread_id;
    int work_by_thread;
} thread_data;

thread_data *thread_data_array;

// END THREAD VARIABLES

// THREAD FUNCTIONS

void update_rgb_buffer(int iteration, int x, int y) {
    int color;

    if (iteration == iteration_max) {
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};

void *compute_mandelbrot_thread(void *threadarg) {
    int work_by_thread, first_buffer_position, work_this_thread;
    long thread_id;

    // EXTRACT DATA FROM THREAD_DATA
    thread_data *my_data;

    my_data = (thread_data *) threadarg;
    thread_id = (long) my_data->thread_id;
    work_by_thread = my_data->work_by_thread;

    // INITIALIZE VARIABLES FOR THREAD
    // Define the first position in the buffer the thread will work
    first_buffer_position = thread_id * work_by_thread;
    
    // As there are equally division of work, some of the final positions of image_buffer
    //   may not be filled up, so this job is assigned to the last thread
    if (thread_id == (num_threads - 1)) {
        work_this_thread = work_by_thread + (image_buffer_size % num_threads);
    } else {
        work_this_thread = work_by_thread;
    }

    // FILL UP IMAGE_BUFFER
    // Each thread will work from first_buffer_position to first_buffer_position + work_this_thread - 1
    for (int i = 0; i < work_this_thread; i++) {
        int iteration;

        // Define the indices according to current buffer position
        int buffer_position = first_buffer_position + i;
        int i_y = buffer_position / i_y_max;
        int i_x = buffer_position % i_x_max;

        double c_y = c_y_min + i_y * pixel_height;
        if (fabs(c_y) < pixel_height / 2) {
            c_y = 0.0;
        };

        double c_x = c_x_min + i_x * pixel_width;

        double z_x = 0.0;
        double z_y = 0.0;

        double z_x_squared = 0.0;
        double z_y_squared = 0.0;

        double escape_radius_squared = 4;

        for (iteration = 0;
            iteration < iteration_max && \
            ((z_x_squared + z_y_squared) < escape_radius_squared);
            iteration++) {
            z_y = 2 * z_x * z_y + c_y;
            z_x = z_x_squared - z_y_squared + c_x;

            z_x_squared = z_x * z_x;
            z_y_squared = z_y * z_y;
        };

        // update_rgb_buffer(iteration, i_x, i_y);
    }

    pthread_exit((void*) thread_id);
};

// END THREAD FUNCTIONS


void allocate_image_buffer() {
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for (int i = 0; i < image_buffer_size; i++) {
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]) {
    if (argc < 7){
        printf("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min c_y_max image_size num_threads\n");
        printf("examples with image_size = 11500 and num_threads = 5:\n");
        printf("    Full Picture:         ./mandelbrot_pth -2.5 1.5 -2.0 2.0 11500 5\n");
        printf("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 0.05 0.15 11500 5\n");
        printf("    Elephant Valley:      ./mandelbrot_pth 0.175 0.375 -0.1 0.1 11500 5\n");
        printf("    Triple Spiral Valley: ./mandelbrot_pth -0.188 -0.012 0.554 0.754 11500 5\n");
        exit(0);
    } else {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &num_threads);

        i_x_max = image_size;
        i_y_max = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width = (c_x_max - c_x_min) / i_x_max;
        pixel_height = (c_y_max - c_y_min) / i_y_max;
    };
};

void write_to_file() {
    FILE *file;
    char *filename = "output.ppm";
    char *comment  = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
            max_color_component_value);

    for (int i = 0; i < image_buffer_size; i++) {
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};

void compute_mandelbrot() {
    // THREAD DEFINITIONS
    pthread_t thread[num_threads];
    pthread_attr_t attr;
    int error_code, work_by_thread;
    long t;
    void *status;

    // INITIALIZE THREADS VARIABLES
    work_by_thread = image_buffer_size / num_threads;
    thread_data_array = malloc(num_threads * sizeof(thread_data));
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // The number of threads cannot be greater than the image size
    if (image_buffer_size < num_threads) {
        num_threads = image_buffer_size;
    }

    // ASSIGN WORK FOR EACH THREADS
    for (t = 0; t < num_threads; t++) {
        // INITIALIZE THREAD DATA
        thread_data_array[t].thread_id = t;
        thread_data_array[t].work_by_thread = work_by_thread;

        // CREATE THREAD
        error_code = pthread_create(&thread[t], &attr, compute_mandelbrot_thread, (void *) &thread_data_array[t]);
        if (error_code){
            printf("ERROR; return code from pthread_create() is %d\n", error_code);
            exit(-1);
        }
    }

    // JOIN THREADS
    pthread_attr_destroy(&attr);

    for (t = 0; t < num_threads; t++){
        error_code = pthread_join(thread[t], &status);
        if (error_code) {
            printf("ERROR; return code from pthread_join() is %d\n", error_code);
            exit(-1);
        };
    };
};

int main(int argc, char *argv[]) {
  init(argc, argv);

  // allocate_image_buffer();

  compute_mandelbrot();

  // write_to_file();

  return 0;
};
