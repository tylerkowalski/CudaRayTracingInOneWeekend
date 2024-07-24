#include "include/colour.cuh"
#include "include/cudaCheck.cuh"
#include "include/ray.cuh"
#include "include/vec3.cuh"

__device__ bool hitSphere(const Point3 &centre, double radius, const Ray &r) {
  // solving the quadratic equation if the ray intersects the sphere
  Vec3 oc = centre - r.origin();
  float a = dot(r.direction(), r.direction());
  float b = -2.0f * dot(r.direction(), oc);
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;
  return discriminant >= 0.0f;
}

__device__ Vec3 colour(const Ray &ray) {
  if (hitSphere(Point3(0, 0, -1), 0.5, ray)) {
    return Colour(1, 0, 0);
  }

  Vec3 unitDirection = unitVector(ray.direction());
  auto alpha = 0.5f * (unitDirection.y() + 1.0f); // puts alpha between 0 and 1
  return (1.0f - alpha) * Colour(1.0, 1.0, 1.0) +
         alpha * Colour(0.5, 0.7,
                        1.0); // linear interpolation between blue and white
}

__global__ void render(Vec3 *fb, int maxX, int maxY, const Vec3 PIXEL00_LOC,
                       const Vec3 pixelDeltaU, const Vec3 pixelDeltaV,
                       const Vec3 cameraCentre) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if ((m > maxY) || (n > maxX))

    return;

  int pixelIndex = m * maxX + n;

  Vec3 pixel = PIXEL00_LOC + (n * pixelDeltaU) + (m * pixelDeltaV);
  Ray ray = Ray(cameraCentre, pixel - cameraCentre);

  fb[pixelIndex] = colour(ray);
}
int main() {
  // image
  auto ASPECT_RATIO = 16.0 / 9.0;
  int IMAGE_WIDTH = 256;

  // calculate the image height with min val = 1
  int IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO);
  IMAGE_HEIGHT = (IMAGE_HEIGHT < 1) ? 1 : IMAGE_HEIGHT;

  // camera
  auto FOCAL_LENGTH = 1.0;
  auto VIEWPORT_HEIGHT = 2.0; // arbitrary
  auto VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (double(IMAGE_WIDTH) / IMAGE_HEIGHT);
  auto CAMERA_CENTRE = Point3(0, 0, 0);

  // vectors that define the viewport
  auto VIEWPORT_U = Vec3(VIEWPORT_WIDTH, 0, 0);
  auto VIEWPORT_V =
      Vec3(0, -VIEWPORT_HEIGHT, 0); // we want the v axis to be going "down"

  // pixel distance vectors
  auto PIXEL_DELTA_U = VIEWPORT_U / IMAGE_WIDTH;
  auto PIXEL_DELTA_V = VIEWPORT_V / IMAGE_HEIGHT;

  // position of upper-left of the viewport
  auto VIEWPORT_UPPER_LEFT = CAMERA_CENTRE - Vec3(0, 0, FOCAL_LENGTH) -
                             (VIEWPORT_U / 2) - (VIEWPORT_V / 2);
  auto PIXEL00_LOC =
      VIEWPORT_UPPER_LEFT + (PIXEL_DELTA_U / 2) + (PIXEL_DELTA_V / 2);

  int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  size_t FB_SIZE = NUM_PIXELS * sizeof(Vec3);

  // allocate frame buffer (using unified memory, which is somewhat cringe)
  Vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, FB_SIZE));

  // NOTE: we want small thread blocks to minimize the probability that 1 thread
  // takes much longer than other threads (the efficiency of the block is
  // limited by the slowest thread) todo : experiment with larger thread blocks
  // NOTE2: we want
  // the number of threads in a block to be a multiple of the warp size
  // (obviously)
  static constexpr int TILE_X = 8;
  static constexpr int TILE_Y = 8;
  dim3 blocks((IMAGE_WIDTH + TILE_X - 1) / TILE_X,
              (IMAGE_HEIGHT + TILE_Y - 1) / TILE_Y);
  // note the fast round up:
  // perfect division -> no change (will round down)
  // non perfect division -> rounds up wrt. original

  dim3 threads(TILE_X, TILE_Y);
  render<<<blocks, threads>>>(fb, IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL00_LOC,
                              PIXEL_DELTA_U, PIXEL_DELTA_V, CAMERA_CENTRE);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // output frame buffer as a ppm image
  std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
  for (int m = IMAGE_HEIGHT - 1; m >= 0;
       --m) { // make m axis (y) point up instead of down
              // since we are reading our m axis in the reverse direction
    for (int n = 0; n < IMAGE_WIDTH; ++n) {
      size_t pixelIndex = m * IMAGE_WIDTH + n;
      writeColour(std::cout, fb[pixelIndex]);
    }
  }
  checkCudaErrors(cudaFree(fb));
}
