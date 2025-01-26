#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct {
  char red;
  char green;
  char blue;
} typedef pixel;

struct {
  char file_type[2];
  int width;
  int height;
  int color_max;
  pixel *data;
} typedef image;

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: ip <input_file_path> <output_file_name>");
    return 1;
  }
  char *input_file_name = argv[1];
  char *output_file_name = argv[2];

  FILE *input_file = fopen(input_file_name, "r");
  if (!input_file) {
    perror("Error opening file");
    return 2;
  }

  FILE *output_file = fopen(output_file_name, "w");
  if (!output_file) {
    perror("Error opening output file");
    return 3;
  }

  image *img = malloc(sizeof(image));
  char *line;
  char *full_header = malloc(sizeof(char) * 64); // for quick output later
  full_header[0] = '\0';
  size_t linelen;

  // read img header info
  getline(&line, &linelen, input_file);
  strcat(full_header, line);
  sscanf(line, "%s", img->file_type);

  getline(&line, &linelen, input_file);
  strcat(full_header, line);
  sscanf(line, "%d %d", &img->width, &img->height);

  getline(&line, &linelen, input_file);
  strcat(full_header, line);
  sscanf(line, "%d", &img->color_max);
  free(line);

  // 3 bytes per pixel * width and height
  img->data = malloc(img->width * sizeof(char) * 3 * img->height);
  int current = 0;

  // read a row (width) at a time (height times)
  fread(img->data, img->width * sizeof(char) * 3, img->height, input_file);

  // img processing (red filter)
  for (int i = 0; i < img->height; i++) {
    for (int j = 0; j < img->width; j++) {
      img->data[i * img->width + j].red = img->color_max;
    }
  }

  // Write header to output
  fwrite(full_header, sizeof(char), strlen(full_header), output_file);

  // Write data to output
  fwrite(img->data, img->width * sizeof(char) * 3, img->height, output_file);

  free(img->data);
  free(img);
  free(full_header);
  return 0;
}
