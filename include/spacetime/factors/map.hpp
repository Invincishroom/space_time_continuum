/*
Taken and adapted from the STEAM-ICP project: https://github.com/utiasASRL/steam_icp
Original author: Yuchen Wu

MIT License

Copyright (c) 2022 ASRL - Autonomous Space Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <queue>
#include <stdexcept>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <tsl/robin_map.h>

namespace steam_icp
{

  // A Point3D
  struct Point3D
  {
    Point3D() = default;
    Point3D(Eigen::Vector3d pt, Eigen::Vector3d normal, double a2d) : pt(pt), normal(normal), a2d(a2d) {}
    Eigen::Vector3d operator-(const Point3D &other) const { return pt - other.pt; }
    Eigen::Vector3d operator-(const Eigen::Vector3d &other) const { return pt - other; }
    bool operator==(const Point3D &other) const { return pt == other.pt; }

    Eigen::Vector3d pt;     // Corrected point taking into account the motion of the sensor during frame acquisition
    Eigen::Vector3d normal; // Normal of the point (if available)
    double a2d = 0.0;       // Planarity coefficient of the point (if available)
  };

  // Voxel
  // Note: coordinates range is in [-32 768, 32 767]
  struct Voxel
  {
    Voxel() = default;

    Voxel(short x, short y, short z) : x(x), y(y), z(z) {}

    bool operator==(const Voxel &vox) const { return x == vox.x && y == vox.y && z == vox.z; }

    inline bool operator<(const Voxel &vox) const
    {
      return x < vox.x || (x == vox.x && y < vox.y) || (x == vox.x && y == vox.y && z < vox.z);
    }

    inline static Voxel coordinates(const Eigen::Vector3d &point, double voxel_size)
    {
      return {short(point.x() / voxel_size), short(point.y() / voxel_size), short(point.z() / voxel_size)};
    }

    short x;
    short y;
    short z;
  };

  using ArrayPoint3D = std::vector<Point3D>;

  struct VoxelBlock
  {
    explicit VoxelBlock(int num_points = 20) : num_points_(num_points) { points.reserve(num_points); }

    bool isFull() const { return num_points_ == static_cast<int>(points.size()); }

    void addPoint(const Point3D &point)
    {
      if (num_points_ < (int)points.size())
        throw std::runtime_error{"voxel is full with size " + std::to_string(points.size())};
      points.push_back(point);
    }

    inline int numPoints() const { return static_cast<int>(points.size()); }

    inline int capacity() { return num_points_; }

    ArrayPoint3D points;

    int life_time = 10;

  private:
    int num_points_;
  };

  using VoxelHashMap = tsl::robin_map<Voxel, VoxelBlock>;

} // namespace steam_icp

// Specialization of std::hash for our custom type Voxel
namespace std
{

  template <>
  struct hash<steam_icp::Voxel>
  {
    std::size_t operator()(const steam_icp::Voxel &vox) const
    {
      const size_t kP1 = 73856093;
      const size_t kP2 = 19349669;
      const size_t kP3 = 83492791;
      return vox.x * kP1 + vox.y * kP2 + vox.z * kP3;
    }
  };

} // namespace std

namespace steam_icp
{

  class Map
  {
  public:
    Map() = default;
    Map(int default_lifetime) : default_lifetime_(default_lifetime) {}

    ArrayPoint3D pointcloud() const
    {
      ArrayPoint3D points;
      points.reserve(size());
      for (auto &voxel : voxel_map_)
      {
        for (int i(0); i < voxel.second.numPoints(); ++i)
          points.push_back(voxel.second.points[i]);
      }
      return points;
    }

    size_t size() const
    {
      size_t map_size(0);
      for (auto &voxel : voxel_map_)
      {
        map_size += (voxel.second).numPoints();
      }
      return map_size;
    }

    void remove(const Eigen::Vector3d &location, double distance)
    {
      std::vector<Voxel> voxels_to_erase;
      for (auto &pair : voxel_map_)
      {
        Point3D pt = pair.second.points[0];
        if ((pt - location).squaredNorm() > (distance * distance))
        {
          voxels_to_erase.push_back(pair.first);
        }
      }
      for (auto &vox : voxels_to_erase)
        voxel_map_.erase(vox);
    }

    void updateAndFilterLifetimes()
    {
      std::vector<Voxel> voxels_to_erase;
      for (VoxelHashMap::iterator it = voxel_map_.begin(); it != voxel_map_.end(); it++)
      {
        auto &voxel_block = (it.value());
        if (!voxel_block.isFull()) // Creates a locking mechanism for voxels that have been sufficiently observed
          voxel_block.life_time -= 1;
        // for (auto &pair : voxel_map_) {
        //   voxel_map_[pair.first].life_time -= 1;
        // it->second.life_time -= 1;
        // if (it->second.life_time <= 0) voxels_to_erase.push_back(it->first);
        if (voxel_block.life_time <= 0)
          voxels_to_erase.push_back(it->first);
      }
      for (auto &vox : voxels_to_erase)
        voxel_map_.erase(vox);
    }
    void clearNonFullVoxels()
    {
      std::vector<Voxel> voxels_to_erase;
      for (VoxelHashMap::iterator it = voxel_map_.begin(); it != voxel_map_.end(); it++)
      {
        auto &voxel_block = (it.value());
        if (!voxel_block.isFull())
          voxels_to_erase.push_back(it->first);
      }
      for (auto &vox : voxels_to_erase)
        voxel_map_.erase(vox);
    }

    void setDefaultLifeTime(int default_lifetime) { default_lifetime_ = default_lifetime; }

    void clear() { voxel_map_.clear(); }

    void add(const Eigen::Vector3d &point, double voxel_size, int max_num_points_in_voxel, double min_distance_points,
             int min_num_points = 0, Eigen::Vector3d normal = Eigen::Vector3d::Zero(), double a2d = 0.0)
    {
      short kx = static_cast<short>(point[0] / voxel_size);
      short ky = static_cast<short>(point[1] / voxel_size);
      short kz = static_cast<short>(point[2] / voxel_size);

      VoxelHashMap::iterator search = voxel_map_.find(Voxel(kx, ky, kz));
      if (search != voxel_map_.end())
      {
        auto &voxel_block = (search.value());

        if (!voxel_block.isFull())
        {
          double sq_dist_min_to_points = 10 * voxel_size * voxel_size;
          for (int i(0); i < voxel_block.numPoints(); ++i)
          {
            auto &_point = voxel_block.points[i];
            double sq_dist = (_point - point).squaredNorm();
            if (sq_dist < sq_dist_min_to_points)
            {
              sq_dist_min_to_points = sq_dist;
            }
          }
          if (sq_dist_min_to_points > (min_distance_points * min_distance_points))
          {
            if (min_num_points <= 0 || voxel_block.numPoints() >= min_num_points)
            {
              voxel_block.addPoint(Point3D(point, normal, a2d));
            }
          }
        }
        voxel_block.life_time = default_lifetime_;
      }
      else
      {
        if (min_num_points <= 0)
        {
          // Do not add points (avoids polluting the map)
          VoxelBlock block(max_num_points_in_voxel);
          block.addPoint(Point3D(point, normal, a2d));
          block.life_time = default_lifetime_;
          voxel_map_[Voxel(kx, ky, kz)] = std::move(block);
        }
      }
    }

    using pair_distance_t = std::tuple<double, Point3D, Voxel>;

    struct Comparator
    {
      bool operator()(const pair_distance_t &left, const pair_distance_t &right) const
      {
        return std::get<0>(left) < std::get<0>(right);
      }
    };

    using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

    ArrayPoint3D searchNeighbors(const Eigen::Vector3d &point, int nb_voxels_visited, double size_voxel_map,
                                 int max_num_neighbors, int threshold_voxel_capacity = 1,
                                 std::vector<Voxel> *voxels = nullptr) const
    {
      if (voxels != nullptr)
        voxels->reserve(max_num_neighbors);

      short kx = static_cast<short>(point[0] / size_voxel_map);
      short ky = static_cast<short>(point[1] / size_voxel_map);
      short kz = static_cast<short>(point[2] / size_voxel_map);

      priority_queue_t priority_queue;

      Voxel voxel(kx, ky, kz);
      for (short kxx = kx - nb_voxels_visited; kxx < kx + nb_voxels_visited + 1; ++kxx)
      {
        for (short kyy = ky - nb_voxels_visited; kyy < ky + nb_voxels_visited + 1; ++kyy)
        {
          for (short kzz = kz - nb_voxels_visited; kzz < kz + nb_voxels_visited + 1; ++kzz)
          {
            voxel.x = kxx;
            voxel.y = kyy;
            voxel.z = kzz;

            auto search = voxel_map_.find(voxel);
            if (search != voxel_map_.end())
            {
              const auto &voxel_block = search.value();
              if (voxel_block.numPoints() < threshold_voxel_capacity)
                continue;
              for (int i(0); i < voxel_block.numPoints(); ++i)
              {
                auto &neighbor = voxel_block.points[i];
                double distance = (neighbor - point).norm();
                if (priority_queue.size() == (size_t)max_num_neighbors)
                {
                  if (distance < std::get<0>(priority_queue.top()))
                  {
                    priority_queue.pop();
                    priority_queue.emplace(distance, neighbor, voxel);
                  }
                }
                else
                  priority_queue.emplace(distance, neighbor, voxel);
              }
            }
          }
        }
      }

      auto size = priority_queue.size();
      ArrayPoint3D closest_neighbors(size);
      if (voxels != nullptr)
      {
        voxels->resize(size);
      }
      for (int i = 0; i < (int)size; ++i)
      {
        closest_neighbors[size - 1 - i] = std::get<1>(priority_queue.top());
        if (voxels != nullptr)
          (*voxels)[size - 1 - i] = std::get<2>(priority_queue.top());
        priority_queue.pop();
      }

      return closest_neighbors;
    }

    struct Neighborhood
    {
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
      double a2D = 1.0; // Planarity coefficient
    };

    static Neighborhood computeNeighborhoodDistribution(const ArrayPoint3D &points)
    {
      if (points[0].a2d != 0.0)
      {
        // Normals are pre-loaded
        Neighborhood neighborhood;
        neighborhood.center = points[0].pt;
        neighborhood.normal = points[0].normal;
        neighborhood.covariance = Eigen::Matrix3d::Identity();
        neighborhood.a2D = points[0].a2d;
        return neighborhood;
      }
      if (points.size() < 3)
      {
        error("Not enough points to compute neighborhood distribution. Are min / max points per voxel set correctly?");
        throw std::runtime_error("Not enough points to compute neighborhood distribution");
      }
      Neighborhood neighborhood;
      // Compute the normals
      Eigen::Vector3d barycenter(Eigen::Vector3d(0, 0, 0));
      for (auto &point : points)
      {
        barycenter += point.pt;
      }
      barycenter /= (double)points.size();
      neighborhood.center = barycenter;

      Eigen::Matrix3d covariance_Matrix(Eigen::Matrix3d::Zero());
      for (auto &point : points)
      {
        for (int k = 0; k < 3; ++k)
          for (int l = k; l < 3; ++l)
            covariance_Matrix(k, l) += (point.pt(k) - barycenter(k)) * (point.pt(l) - barycenter(l));
      }
      covariance_Matrix(1, 0) = covariance_Matrix(0, 1);
      covariance_Matrix(2, 0) = covariance_Matrix(0, 2);
      covariance_Matrix(2, 1) = covariance_Matrix(1, 2);
      neighborhood.covariance = covariance_Matrix;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_Matrix);
      Eigen::Vector3d normal(es.eigenvectors().col(0).normalized());
      neighborhood.normal = normal;

      // Compute planarity from the eigen values
      double sigma_1 = sqrt(std::abs(es.eigenvalues()[2])); // Be careful, the eigenvalues are not correct with the
                                                            // iterative way to compute the covariance matrix
      double sigma_2 = sqrt(std::abs(es.eigenvalues()[1]));
      double sigma_3 = sqrt(std::abs(es.eigenvalues()[0]));
      neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1;

      if (neighborhood.a2D != neighborhood.a2D)
      {
        std::cout << "[ERROR] FOUND NAN!!!";
        throw std::runtime_error("error");
      }

      return neighborhood;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    // Subsample to keep one (random) point in every voxel of the current frame
    // Run std::shuffle() first in order to retain a random point for each voxel.
    static void subSampleFrame(std::vector<Point3D> &frame, double size_voxel)
    {
      std::unordered_map<Voxel, std::vector<Point3D>> grid;
      for (int i = 0; i < (int)frame.size(); i++)
      {
        auto kx = static_cast<short>(frame[i].pt[0] / size_voxel);
        auto ky = static_cast<short>(frame[i].pt[1] / size_voxel);
        auto kz = static_cast<short>(frame[i].pt[2] / size_voxel);
        grid[Voxel(kx, ky, kz)].push_back(frame[i]);
      }
      frame.resize(0);
      int step = 0; // to take one random point inside each voxel (but with identical results when lunching the SLAM a
                    // second time)
      for (const auto &n : grid)
      {
        if (n.second.size() > 0)
        {
          // frame.push_back(n.second[step % (int)n.second.size()]);
          frame.push_back(n.second[0]);
          step++;
        }
      }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    static void gridSampling(const std::vector<Point3D> &frame, std::vector<Point3D> &keypoints, double size_voxel_subsampling)
    {
      keypoints.resize(0);
      std::vector<Point3D> frame_sub;
      frame_sub.resize(frame.size());
      for (int i = 0; i < (int)frame_sub.size(); i++)
      {
        frame_sub[i] = frame[i];
      }
      subSampleFrame(frame_sub, size_voxel_subsampling);
      keypoints.reserve(frame_sub.size());
      for (int i = 0; i < (int)frame_sub.size(); i++)
      {
        keypoints.push_back(frame_sub[i]);
      }
    }

    void save(const std::string &filename) const
    {
      FILE *pFile = fopen(filename.c_str(), "w");
      if (pFile == nullptr)
      {
        throw std::runtime_error("Could not open file " + filename + " for writing.");
      }
      if (voxel_map_.empty())
      {
        fclose(pFile);
        return;
      }
      ArrayPoint3D points = pointcloud();
      for (const auto &point : points)
      {
        fprintf(pFile, "%f,%f,%f\n", point.pt.x(), point.pt.y(), point.pt.z());
      }
      fclose(pFile);
    }

    void saveCheckpoint(double time)
    {
      ArrayPoint3D points;
      points.reserve(size());
      for (auto &voxel : voxel_map_)
      {
        if (!voxel.second.isFull())
          continue;
        for (int i(0); i < voxel.second.numPoints(); ++i)
          points.push_back(voxel.second.points[i]);
      }

      map_history.push_back(std::pair<double, ArrayPoint3D>(time, points));
    }

    ArrayPoint3D getPointCloudAtTime(double time)
    {
      if (map_history.empty())
      {
        return ArrayPoint3D();
      }

      auto closest_it = map_history.begin();
      double min_diff = std::abs(closest_it->first - time);

      for (auto it = map_history.begin(); it != map_history.end(); ++it)
      {
        double diff = std::abs(it->first - time);
        if (diff < min_diff)
        {
          min_diff = diff;
          closest_it = it;
        }
      }

      return closest_it->second;
    }

  private:
    VoxelHashMap voxel_map_;
    int default_lifetime_ = 10;
    std::vector<std::pair<double, ArrayPoint3D>> map_history;
  };

} // namespace steam_icp
