#ifndef Vec3_H
#define Vec3_H

#include <cmath>
#include <iostream>
#include <ostream>

using std::sqrt;

class Vec3 {
public:
  float e[3];

  __host__ __device__ Vec3() : e{0, 0, 0} {}
  __host__ __device__ Vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

  __host__ __device__ float x() const { return e[0]; }
  __host__ __device__ float y() const { return e[1]; }
  __host__ __device__ float z() const { return e[2]; }

  __host__ __device__ Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ float operator[](int i) const { return e[i]; }
  __host__ __device__ float &operator[](int i) { return e[i]; }

  __host__ __device__ Vec3 &operator+=(const Vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  __host__ __device__ Vec3 &operator*=(float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  __host__ __device__ Vec3 &operator/=(float t) { return *this *= 1 / t; }

  __host__ __device__ float lengthSquared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  __host__ __device__ float length() const { return sqrt(lengthSquared()); }
};

// probably not the best for type-safety, but we will use it for geometric
// clarity
using Point3 = Vec3;

// vector utility functions
inline std::ostream &operator<<(std::ostream &out, const Vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
  return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
  return t * v;
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v, float t) {
  return (1 / t) * v;
}

__host__ __device__ inline double dot(const Vec3 &u, const Vec3 &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline Vec3 unitVector(const Vec3 &v) {
  return v / v.length();
}

#endif