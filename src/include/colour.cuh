#ifndef COLOUR_CUH
#define COLOUR_CUH

#include "vec3.cuh"

#include <iostream>

// note that we are using vec3 for both coloUrs and geometry
using Colour = Vec3;

void writeColour(std::ostream &out, const Colour &pixelColour) {
  auto r = pixelColour.x();
  auto g = pixelColour.y();
  auto b = pixelColour.z();

  // translate values from [0,1] -> [0,255]
  int rbyte = int(255.999 * r);
  int gbyte = int(255.999 * g);
  int bbyte = int(255.999 * b);

  // write out the components as defined in PPM format
  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif