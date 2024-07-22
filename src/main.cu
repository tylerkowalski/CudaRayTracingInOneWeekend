#include "include/cudaCheck.cuh"
#include "include/vec3.cuh"

__global__ void render(Vec3 *fb, int maxX, int maxY) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if ((m > maxY) || (n > maxX))
    return;

  int pixelIndex = m * maxX + n;

  // get more red and green has we get to larger pixel numbers
  fb[pixelIndex] = Vec3(float(n) / maxX, float(m) / maxY, 0.2);

  // fb[pixelIndex + 0] = float(n) / maxX;
  // fb[pixelIndex + 1] = float(m) / maxY;
  // fb[pixelIndex + 2] = 0.2;
}

int main() {
  // image
  static constexpr int IMAGE_WIDTH = 256;
  static constexpr int IMAGE_HEIGHT = 256;

  static constexpr int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  static constexpr size_t FB_SIZE = NUM_PIXELS * sizeof(Vec3);

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
  render<<<blocks, threads>>>(fb, IMAGE_WIDTH, IMAGE_HEIGHT);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // output frame buffer as a ppm image
  std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
  for (int m = IMAGE_HEIGHT - 1; m >= 0;
       --m) { // make m axis (y) point up instead of down
              // since we are reading our m axis in the reverse direction
    for (int n = 0; n < IMAGE_WIDTH; ++n) {
      size_t pixelIndex = m * IMAGE_WIDTH + n;

      auto r = fb[pixelIndex].x();
      auto g = fb[pixelIndex].y();
      auto b = fb[pixelIndex].z();

      int ir = int(255.999 * r);
      int ig = int(255.999 * g);
      int ib = int(255.999 * b);

      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
  checkCudaErrors(cudaFree(fb));
}
