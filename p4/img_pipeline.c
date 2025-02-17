/**
 * Isaac Denny
 * Feb 6 2025
 * CSC4310
 *
 * Greyscale filters and horizontally flips a PPM image in
 * serial and parallel.
 *
 * Compile:  gcc img_pipeline.c -o ip -fopenmp -lpthread -Wall
 * Run: ./ip <input_img> <output_name NO EXTENSION>
 */
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/select.h>
#include <unistd.h>
#include <sys/time.h>

struct {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
} typedef pixel;

struct {
  char type[3];
  int width;
  int height;
  int max_color;
  pixel **data;
} typedef image;

/**
 * Reads a P6 image file to struct, allocates necessary memory.
 * Does not close the image
 * pre: open file, empty struct
 * post: open file, filled image struct
 */
void read_img(FILE *f, image *img);

/**
 * Writes a P6 image struct to an output file.
 * Does not close the file or free the struct memory
 * pre: open file, filled struct
 * post: data written to open file, filled struct
 */
void write_img(FILE *f, image *img);

/**
 * Flips one row of pixels across the vertical axis
 * pre: original image
 * post: specified row is flipped
 */
void flip_row(image *img, pixel *row);

/**
 * Greyscales one row of pixels according to the luminosity method
 * pre: original image
 * post: specified row is greyscaled
 */
void greyscale_row(image *img, pixel *row);

/**
 * Processes an image with the above filters serially
 * pre: original image
 * post: filters are applied to image
 */
void serial_process_img(image *img);

/**
 * Processes an image with the above filters using
 * a parallel pipeline
 * pre: original image
 * post: filters are applied to image
 */
void pipeline_process_img(image *img);

/**
 * Helper for duplicating images in memory
 * pre: src image is allocated and filled
 * post: new image struct allocated and returned
 */
image *imgdup(image *src);

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: ip <input_img> <output_name NO EXTENSION>\n");
    return 1;
  }

  char *input_file_name = argv[1];
  char *output_file_name = argv[2];

  // need 2 output files for serial and pipeline (mostly for testing)
  char *out_file_name1 =
      malloc(strlen(output_file_name) + 2); // 6 bytes for "_s.ppm" (serial)
  char *out_file_name2 =
      malloc(strlen(output_file_name) + 2); // 6 bytes for "_p.ppm" (pipeline)
  sprintf(out_file_name1, "%s_s.ppm", output_file_name);
  sprintf(out_file_name2, "%s_p.ppm", output_file_name);

  FILE *in_file = fopen(input_file_name, "r");
  if (in_file == NULL) {
    perror("Error opening input file");
    return 2;
  }

  FILE *out_file_s = fopen(out_file_name1, "w");
  if (out_file_s == NULL) {
    perror("Error opening serial output file");
    return 3;
  }

  FILE *out_file_p = fopen(out_file_name2, "w");
  if (out_file_p == NULL) {
    perror("Error opening pipeline output file");
    return 4;
  }

  struct timeval t1, t2;
  double runtime;

  image *img = malloc(sizeof(image));
  bzero(img, sizeof(image));
  read_img(in_file, img);

  // copy so we can reuse
  image *cp1 = imgdup(img);
  gettimeofday(&t1, NULL);
  serial_process_img(cp1);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1.e6;
  printf("Time to serial process: %lf\n", runtime); 

  gettimeofday(&t1, NULL);
  pipeline_process_img(img);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1.e6;
  printf("Time to parallel process: %lf\n", runtime); 

  // structs
  write_img(out_file_s, cp1);
  write_img(out_file_p, img);

  free(img->data);
  free(img);
  free(cp1->data);
  free(cp1);

  fclose(in_file);
  fclose(out_file_p);
  fclose(out_file_s);
  return 0;
}

image *imgdup(image *src) {
  image *dup = malloc(sizeof(image));                // new image
  dup->data = (pixel**)malloc(src->height * sizeof(pixel *)); // rows
  pixel *temp = (pixel*)malloc(src->width * src->height * sizeof(pixel));

  // copy data to temp for one write
  memcpy(temp, src->data[0], src->width * src->height * sizeof(pixel));
  for (int i = 0; i < src->height; i++) {
    dup->data[i] = &temp[i * src->width];
  }

  // copy headers over
  dup->width = src->width;
  dup->height = src->height;
  dup->max_color = src->max_color;
  memcpy(dup->type, src->type, 3);

  return dup;
}

void serial_process_img(image *img) {
  for (int i = 0; i < img->height; i++) {
    greyscale_row(img, img->data[i]);
  }
  for (int i = 0; i < img->height; i++) {
    flip_row(img, img->data[i]);
  }
}

void pipeline_process_img(image *img) {
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    perror("Error creating pipe");
    exit(4);
  }

  int j = -2; // -1 is reserved
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) {
      // greyscale a row, then pass it to the next thread
      for (int i = 0; i < img->height; i++) {
        // printf("  Thread %d: greyscaling row %d\n", omp_get_thread_num(), i);
        greyscale_row(img, img->data[i]);
        write(pipefd[1], &i, sizeof(int));
      }

      int h = -1; // signal to next thread that it's finished
      write(pipefd[1], &h, sizeof(int));
    } else {
      // read from pipe, if an index is there, flip the row
      for (;;) {
        read(pipefd[0], &j, sizeof(int));
        if (j == -1) {
          // this is the previous thread telling us we're done
          break;
        }
        /* this might be nice here but could slow us down
          if (j < 0 || j > img->height) {
            fprintf(stderr, "Error: invalid index from pipe %d", j);
            continue;
          }
        */
        // printf("  Thread %d: flipping row %d\n", omp_get_thread_num(), j);
        flip_row(img, img->data[j]);
      }
    }
  }

  close(pipefd[0]);
  close(pipefd[1]);
}

void read_img(FILE *f, image *img) {
  char *line;
  size_t line_len;
  getline(&line, &line_len, f);

  // Handle img type
  sscanf(line, "%2c", &(img->type[0]));

  // Handle comments
  while (getline(&line, &line_len, f)) {
    if (line[0] != '#') {
      break;
    }
  }

  // read the rest of the header (width, height, max_color)
  sscanf(line, "%d %d", &(img->width), &(img->height));
  getline(&line, &line_len, f);
  sscanf(line, "%d", &(img->max_color));

  // read image data and reorganize
  img->data = (pixel **)malloc(img->height * sizeof(pixel *));
  pixel *temp = (pixel *)malloc(img->width * img->height * sizeof(pixel));
  fread(temp, sizeof(unsigned char), img->width * img->height * sizeof(pixel),
        f);

  for (int i = 0; i < img->height; i++) {
    img->data[i] = &temp[i * img->width];
  }
}

void write_img(FILE *f, image *img) {
  fprintf(f, "%s\n%d %d\n%d\n", img->type, img->width, img->height,
          img->max_color);
  fwrite(img->data[0], sizeof(unsigned char),
         img->width * img->height * sizeof(pixel), f);
}

void flip_row(image *img, pixel *row) {
  for (int i = 0; i < img->width / 2; i++) {
    pixel temp = row[i];
    row[i] = row[img->width - i];
    row[img->width - i] = temp;
  }
}

void greyscale_row(image *img, pixel *row) {
  for (int i = 0; i < img->width; i++) {
    int greyvalue =
        0.21 * row[i].red + 0.72 * row[i].green + 0.07 * row[i].blue;
    row[i].red = greyvalue;
    row[i].green = greyvalue;
    row[i].blue = greyvalue;
  }
}
