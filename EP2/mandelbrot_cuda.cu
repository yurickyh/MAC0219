#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int image_size;

int i_x_max;
int i_y_max;
int image_buffer_size;

void print_error_message() {
  printf(
      "usage: ./mandelbrot_seq c_x_min c_x_max c_y_min c_y_max image_size\n");
  printf("examples with image_size = 11500:\n");
  printf(
      "    Full Picture:         ./mandelbrot_seq -2.5 1.5 -2.0 2.0 11500\n");
  printf(
      "    Seahorse Valley:      ./mandelbrot_seq -0.8 -0.7 0.05 0.15 "
      "11500\n");
  printf(
      "    Elephant Valley:      ./mandelbrot_seq 0.175 0.375 -0.1 0.1 "
      "11500\n");
  printf(
      "    Triple Spiral Valley: ./mandelbrot_seq -0.188 -0.012 0.554 0.754 "
      "11500\n");
  exit(0);
};

// void write_to_file() {
//   FILE *file;
//   char *filename = "output.ppm";
//   char *comment = "# ";

//   int max_color_component_value = 255;

//   file = fopen(filename, "wb");

//   fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
//           max_color_component_value);

//   for (int i = 0; i < image_buffer_size; i++) {
//     fwrite(image_buffer[i], 1, 3, file);
//   };

//   fclose(file);
// };

__global__ 
void compute_mandelbrot(float* image_buffer, int gradient_size, int pixel_height,
                        int pixel_width, int i_x_max, int i_y_max, int iteration_max,
                        double c_x_min, double c_x_max, double c_y_min, double c_y_max) {
  double z_x;
  double z_y;
  double z_x_squared;
  double z_y_squared;
  double escape_radius_squared = 4;

  int iteration;
  int i_x;
  int i_y;

  double c_x;
  double c_y;

  int colors[17][3] = {
    {66, 30, 15},    {25, 7, 26},     {9, 1, 47},      {4, 4, 73},
    {0, 7, 100},     {12, 44, 138},   {24, 82, 177},   {57, 125, 209},
    {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95},
    {255, 170, 0},   {204, 128, 0},   {153, 87, 0},    {106, 52, 3},
    {16, 16, 16},
  };

  for (i_y = 0; i_y < i_y_max; i_y++) {
    c_y = c_y_min + i_y * pixel_height;

    if (fabs(c_y) < pixel_height / 2) {
      c_y = 0.0;
    };

    for (i_x = 0; i_x < i_x_max; i_x++) {
      c_x = c_x_min + i_x * pixel_width;

      z_x = 0.0;
      z_y = 0.0;

      z_x_squared = 0.0;
      z_y_squared = 0.0;

      for (iteration = 0; iteration < iteration_max &&
                          ((z_x_squared + z_y_squared) < escape_radius_squared);
           iteration++) {
        z_y = 2 * z_x * z_y + c_y;
        z_x = z_x_squared - z_y_squared + c_x;

        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
      };

      int color;

      if (iteration == iteration_max) {
        image_buffer[(i_y_max * i_y) + i_x] = colors[gradient_size][0];
        image_buffer[(i_y_max * i_y) + i_x+1] = colors[gradient_size][1];
        image_buffer[(i_y_max * i_y) + i_x+2] = colors[gradient_size][2];
      } else {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * i_y) + i_x] = colors[color][0];
        image_buffer[(i_y_max * i_y) + i_x+1] = colors[color][1];
        image_buffer[(i_y_max * i_y) + i_x+2] = colors[color][2];
      };
      // if (iteration == iteration_max) {
      //   image_buffer[(i_y_max * i_y) + i_x][0] = colors[gradient_size][0];
      //   image_buffer[(i_y_max * i_y) + i_x][1] = colors[gradient_size][1];
      //   image_buffer[(i_y_max * i_y) + i_x][2] = colors[gradient_size][2];
      // } else {
      //   color = iteration % gradient_size;

      //   image_buffer[(i_y_max * i_y) + i_x][0] = colors[color][0];
      //   image_buffer[(i_y_max * i_y) + i_x][1] = colors[color][1];
      //   image_buffer[(i_y_max * i_y) + i_x][2] = colors[color][2];
      // };
    };
  };
};

int main(int argc, char *argv[]) {
  if (argc < 6) {
    print_error_message();
  } else {
    sscanf(argv[1], "%lf", &c_x_min);
    sscanf(argv[2], "%lf", &c_x_max);
    sscanf(argv[3], "%lf", &c_y_min);
    sscanf(argv[4], "%lf", &c_y_max);
    sscanf(argv[5], "%d", &image_size);

    i_x_max = image_size;
    i_y_max = image_size;
    image_buffer_size = image_size * image_size;

    pixel_width = (c_x_max - c_x_min) / i_x_max;
    pixel_height = (c_y_max - c_y_min) / i_y_max;
  };

  float *image_buffer;
  int iteration_max = 200;
  int gradient_size = 16;

  int rgb_size = 3;

  cudaMallocManaged(&image_buffer, (image_buffer_size * rgb_size)*sizeof(float));

  compute_mandelbrot<<<1, 1>>>(image_buffer, gradient_size, pixel_height, pixel_width, 
                     i_x_max, i_y_max, iteration_max, c_x_min, c_x_max, c_y_min,
                     c_y_max);

  cudaDeviceSynchronize();

  cout << image_buffer[20] << endl;

  // write_to_file();

  // FILE *file;
  // char *filename = "output.ppm";
  // char *comment = "# ";

  // int max_color_component_value = 255;

  // file = fopen(filename, "wb");

  // fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
  //         max_color_component_value);

  // for (int i = 0; i < image_buffer_size; i++) {
  //   fwrite(image_buffer[i], 1, 3, file);
  // };

  // fclose(file);

  return 0;
};
