#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define MASTER 0

// --------------- OMPI VARIABLES ---------------

int num_tasks;
int task_id;
char hostname[MPI_MAX_PROCESSOR_NAME];

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
    {66, 30, 15},
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3},
    {16, 16, 16},
};

typedef struct
{
  int x;
  int y;
  int iteration;
} rgb_data;

void update_rgb_buffer(int iteration, int x, int y)
{
  int color;

  if (iteration == iteration_max)
  {
    image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
    image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
    image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
  }
  else
  {
    color = iteration % gradient_size;

    image_buffer[(i_y_max * y) + x][0] = colors[color][0];
    image_buffer[(i_y_max * y) + x][1] = colors[color][1];
    image_buffer[(i_y_max * y) + x][2] = colors[color][2];
  };
};

// END THREAD FUNCTIONS

void allocate_image_buffer()
{
  int rgb_size = 3;
  image_buffer = (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);

  for (int i = 0; i < image_buffer_size; i++)
  {
    image_buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
  };
};

void init(int argc, char *argv[])
{
  if (argc < 6)
  {
    printf("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min c_y_max image_size num_threads\n");
    printf("examples with image_size = 11500 and num_threads = 5:\n");
    printf("    Full Picture:         ./mandelbrot_pth -2.5 1.5 -2.0 2.0 11500 5\n");
    printf("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 0.05 0.15 11500 5\n");
    printf("    Elephant Valley:      ./mandelbrot_pth 0.175 0.375 -0.1 0.1 11500 5\n");
    printf("    Triple Spiral Valley: ./mandelbrot_pth -0.188 -0.012 0.554 0.754 11500 5\n");
    exit(0);
  }
  else
  {
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
};

void write_to_file()
{

  FILE *file;
  char *filename = "output.ppm";
  char *comment = "# ";

  int max_color_component_value = 255;

  file = fopen(filename, "wb");

  fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
          max_color_component_value);

  for (int i = 0; i < image_buffer_size; i++)
  {
    fwrite(image_buffer[i], 1, 3, file);
  };

  fclose(file);
};

int get_number_slaves()
{
  if ((num_tasks - 1) > image_buffer_size)
    return image_buffer_size;
  return (num_tasks - 1);
}

int get_work_amount(int work_by_task, int num_slaves)
{
  if (task_id == num_slaves)
    return work_by_task + (image_buffer_size % num_slaves);
  return work_by_task;
}

void create_struct()
{
  
}

void compute_mandelbrot()
{
  MPI_Status status;
  int num_slaves = get_number_slaves();

  int work_by_task = image_buffer_size / num_slaves;
  int work_this_task = get_work_amount(work_by_task, num_slaves);

  rgb_data rgb_data_array[work_this_task];
  MPI_Datatype rgb_data_type;

  
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
  int blockcounts[3] = {1, 1, 1};
  MPI_Aint offsets[3];

  offsets[0] = offsetof(rgb_data, x);
  offsets[1] = offsetof(rgb_data, y);
  offsets[2] = offsetof(rgb_data, iteration);

  // Finally create the type.
  MPI_Type_create_struct(3, blockcounts, offsets, types, &rgb_data_type);
  MPI_Type_commit(&rgb_data_type);

  if (task_id == MASTER)
  {
    allocate_image_buffer();

    for (int i = 1; i <= num_slaves; i++)
    {
      MPI_Send(&i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    for (int i = 1; i <= num_slaves; i++)
    { 
      MPI_Recv(rgb_data_array, work_this_task, rgb_data_type, i, 1, MPI_COMM_WORLD, &status);

      for (int j = 0; j < work_this_task; j++)
      {
        update_rgb_buffer(rgb_data_array[j].iteration, rgb_data_array[j].x, rgb_data_array[j].y);
      }
    }

    write_to_file();
  }
  else
  {
    int dummy;
    MPI_Recv(&dummy, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    for (int i = 0; i < work_this_task; i++)
    {
      int iteration;

      // Define the indices according to current buffer position
      int first_buffer_position = (task_id - 1) * work_by_task;
      int buffer_position = first_buffer_position + i;
      int i_y = buffer_position / i_y_max;
      int i_x = buffer_position % i_x_max;

      double c_y = c_y_min + i_y * pixel_height;
      if (fabs(c_y) < pixel_height / 2)
      {
        c_y = 0.0;
      };

      double c_x = c_x_min + i_x * pixel_width;

      double z_x = 0.0;
      double z_y = 0.0;

      double z_x_squared = 0.0;
      double z_y_squared = 0.0;

      double escape_radius_squared = 4;

      for (iteration = 0;
           iteration < iteration_max &&
           ((z_x_squared + z_y_squared) < escape_radius_squared);
           iteration++)
      {
        z_y = 2 * z_x * z_y + c_y;
        z_x = z_x_squared - z_y_squared + c_x;

        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
      };

      rgb_data_array[i].x = i_x;
      rgb_data_array[i].y = i_y;
      rgb_data_array[i].iteration = iteration;
    }

    MPI_Send(rgb_data_array, work_this_task, rgb_data_type, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Finalize();
};

int main(int argc, char *argv[])
{
  // MPI_Init(&argc, &argv);
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  init(argc, argv);

  compute_mandelbrot();

  return 0;
};
