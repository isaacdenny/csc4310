#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

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

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: ip <input_img> <output_img>\n");
    return 1;
  }
  char *input_file_name = argv[1];
  char *output_file_name = argv[2];

  FILE *in_file = fopen(input_file_name, "r");
  if (in_file == NULL) {
    perror("Error opening input file");
    return 2;
  }
  FILE *out_file = fopen(output_file_name, "w");
  if (out_file == NULL) {
    perror("Error opening output file");
    return 3;
  }

  image *img = malloc(sizeof(image));
  bzero(img, sizeof(image));
  read_img(in_file, img);

  for (int i = 0; i < img->height; i++) {
    greyscale_row(img, img->data[i]);
  }
  for (int i = 0; i < img->height; i++) {
    flip_row(img, img->data[i]);
  }
  // structs
  write_img(out_file, img);

  free(img->data);
  free(img);
  return 0;
}

void read_img(FILE *f, image *img) {
  char *line;
  size_t line_len;
  getline(&line, &line_len, f);

  // Handle img type
  sscanf(line, "%2c", &(img->type[0]));
  printf("Type: %s\n", img->type);

  // Handle comments
  while (getline(&line, &line_len, f)) {
    if (line[0] != '#') {
      printf("not a comment: %s\n", line);
      break;
    }

    printf("Comment: %s\n", line);
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
  fwrite(img->data[0], sizeof(unsigned char), img->width * img->height * sizeof(pixel), f);
}

void flip_row(image* img, pixel* row) {
  for (int i = 0; i < img->width / 2; i++) {
    pixel temp = row[i];
    row[i] = row[img->width - i];
    row[img->width - i] = temp;
  }
}

void greyscale_row(image* img, pixel* row) {
  for (int i = 0; i < img->width; i++) {
    int greyvalue = 0.21 * row[i].red + 0.72 * row[i].green + 0.07 * row[i].blue;
    row[i].red = greyvalue;
    row[i].green = greyvalue;
    row[i].blue = greyvalue;
  }
}
