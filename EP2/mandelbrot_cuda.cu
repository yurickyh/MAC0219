#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void print_instructions() {
  printf(
      "usage: ./mandelbrot_cuda c_x_min c_x_max c_y_min c_y_max image_size\n");
  printf("examples with image_size = 4096 and num_threads = 256:\n");
  printf(
      "    Full Picture:         ./mandelbrot_cuda -2.5 1.5 -2.0 2.0 4096 256\n");
  printf(
      "    Seahorse Valley:      ./mandelbrot_cuda -0.8 -0.7 0.05 0.15 "
      "4096 256\n");
  printf(
      "    Elephant Valley:      ./mandelbrot_cuda 0.175 0.375 -0.1 0.1 "
      "4096 256\n");
  printf(
      "    Triple Spiral Valley: ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 "
      "4096 256\n");
  exit(0);
};

void write_to_file(unsigned char *image_buffer, int i_x_max, int i_y_max,
                   int image_buffer_size) {
  FILE *file;
  const char *filename = "output.ppm";
  const char *comment = "# ";

  int max_color_component_value = 255;

  file = fopen(filename, "wb");

  fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
          max_color_component_value);

  for (int i = 0; i < image_buffer_size; i++) {
    unsigned char buffer[3] = {image_buffer[i],
                              image_buffer[i + image_buffer_size],
                              image_buffer[i + (image_buffer_size * 2)]};
    fwrite(buffer, 1, 3, file);
  };

  fclose(file);
};

__global__ 
void compute_mandelbrot(unsigned char *d_image_buffer, int gradient_size,
                        int iteration_max, double c_x_min, double c_x_max,
                        double c_y_min, double c_y_max, int image_buffer_size,
                        int i_x_max, int i_y_max, double pixel_width,
                        double pixel_height) {
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

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (i_y = index; i_y < i_y_max; i_y += stride) {
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

      if (iteration == iteration_max) {
        d_image_buffer[(i_y_max * i_y) + i_x] = colors[16][0];
        d_image_buffer[(i_y_max * i_y) + i_x + image_buffer_size] =
            colors[16][1];
        d_image_buffer[(i_y_max * i_y) + i_x + (2 * image_buffer_size)] =
            colors[16][2];
      } else {
        int color = iteration % 16;
        d_image_buffer[(i_y_max * i_y) + i_x] = colors[color][0];
        d_image_buffer[(i_y_max * i_y) + i_x + image_buffer_size] =
            colors[color][1];
        d_image_buffer[(i_y_max * i_y) + i_x + (2 * image_buffer_size)] =
            colors[color][2];
      };
    };
  };
};

int main(int argc, char *argv[]) {
  if (argc < 7) {
    print_instructions();
    return 0;
  }

  double c_x_min;
  double c_x_max;
  double c_y_min;
  double c_y_max;

  int image_size;
  int num_threads;
  unsigned char *image_buffer;
  unsigned char *d_image_buffer;

  int gradient_size = 16;
  int iteration_max = 200;

  sscanf(argv[1], "%lf", &c_x_min);
  sscanf(argv[2], "%lf", &c_x_max);
  sscanf(argv[3], "%lf", &c_y_min);
  sscanf(argv[4], "%lf", &c_y_max);
  sscanf(argv[5], "%d", &image_size);
  sscanf(argv[6], "%d", &num_threads);

  int i_x_max = image_size;
  int i_y_max = image_size;

  int image_buffer_size = image_size * image_size;

  double pixel_width = (c_x_max - c_x_min) / i_x_max;
  double pixel_height = (c_y_max - c_y_min) / i_y_max;

  int rgb_size = 3;
  image_buffer = (unsigned char *)malloc(sizeof(unsigned char) *
                                         image_buffer_size * rgb_size);

  cudaMalloc(&d_image_buffer, sizeof(unsigned char) * image_buffer_size * rgb_size);

  int blockSize = num_threads;
  int numBlocks = (image_buffer_size + blockSize - 1) / blockSize;

  compute_mandelbrot<<<numBlocks, blockSize>>>(d_image_buffer, gradient_size,
                     iteration_max, c_x_min, c_x_max, c_y_min, c_y_max,
                     image_buffer_size, i_x_max, i_y_max, pixel_width,
                     pixel_height);

  cudaDeviceSynchronize();

  cudaMemcpy(image_buffer, d_image_buffer, 
             sizeof(unsigned char) * image_buffer_size * rgb_size, 
             cudaMemcpyDeviceToHost);

  write_to_file(image_buffer, i_x_max, i_y_max, image_buffer_size);

  return 0;
};
