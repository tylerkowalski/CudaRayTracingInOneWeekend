#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray {
private:
  Point3 orig;
  Vec3 dir;

public:
  __device__ Ray() {}

  __device__ Ray(const Point3 &origin, const Vec3 &direction)
      : orig{origin}, dir{direction} {}

  __device__ const Point3 &origin() const { return orig; }
  __device__ const Vec3 &direction() const { return dir; }

  __device__ Point3 at(float t) const { return orig + t * dir; }
};

#endif