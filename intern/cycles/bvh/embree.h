/* SPDX-FileCopyrightText: 2018-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_EMBREE

#  include <embree4/rtcore.h>
#  include <embree4/rtcore_scene.h>

#  include "bvh/bvh.h"
#  include "bvh/params.h"

#  include "util/types_float3.h"
#  include "util/unique_ptr.h"
#  include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Hair;
class Mesh;
class PointCloud;

/* Prism shell (research prototype, see PRIMITIVE_PRISM in kernel/types.h):
 * host-only data for the per-triangle bounds callback (add_prism_shell /
 * prism_bounds_func in embree.cpp). The intersect callback itself
 * (kernel/device/cpu/bvh.h) does NOT use this -- it re-derives everything
 * from kernel_data_fetch (triangle_vertices, the object's
 * prism_shell_thickness) the same way real triangle hits do, which is both
 * GPU-portable-style and avoids having to keep a second struct layout in
 * sync with this one. This struct only needs to be host-visible. Owned by
 * BVHEmbree::prism_geom_data so the pointer handed to Embree stays valid
 * for the geometry's lifetime.
 *
 * `triangles`/`verts` are resolved ONCE, in add_prism_shell(), rather than
 * storing a `Mesh*` and re-resolving them (mesh->get_triangles().data(),
 * mesh->attributes.find(ATTR_STD_POSITION)) on every prism_bounds_func
 * call -- Embree invokes that callback once per triangle during its own
 * (already-parallel) BVH build, so a per-call attribute-table lookup was
 * pure O(triangles) redundant work for data that's identical every time
 * (the real triangle mesh doesn't have this problem at all: Embree reads
 * its vertex buffer directly via a shared pointer, no per-primitive
 * callback needed). */
struct BVHEmbreePrismUserData {
  const int *triangles;
  const packed_float3 *verts;
  float thickness;
};

class BVHEmbree : public BVH {
 public:
  void build(Progress &progress,
             Stats *stats,
             RTCDevice rtc_device,
             const bool rtc_device_is_sycl_ = false);
  void refit(Progress &progress);

#  if defined(WITH_EMBREE_GPU) && RTC_VERSION >= 40302
  RTCError offload_scenes_to_gpu(const vector<RTCScene> &scenes);
#  endif

  const char *get_error_string(RTCError error_code);

  RTCScene scene;

  BVHEmbree(const BVHParams &params,
            const vector<Geometry *> &geometry,
            const vector<Object *> &objects);
  ~BVHEmbree() override;

 protected:
  void add_object(Object *ob, const int i);
  void add_instance(Object *ob, const int i);
  void add_curves(const Object *ob, const Hair *hair, const int i);
  void add_points(const Object *ob, const PointCloud *pointcloud, const int i);
  void add_gsplats(const Object *ob, const PointCloud *pointcloud, const int i);
  void add_triangles(const Object *ob, const Mesh *mesh, const int i);
  void add_prism_shell(const Object *ob, const Mesh *mesh, const int i);

 private:
  vector<unique_ptr<BVHEmbreePrismUserData>> prism_geom_data;

  void set_tri_vertex_buffer(RTCGeometry geom_id, const Mesh *mesh, const bool update);
  void set_curve_vertex_buffer(RTCGeometry geom_id, const Hair *hair, const bool update);
  void set_point_vertex_buffer(RTCGeometry geom_id,
                               const PointCloud *pointcloud,
                               const bool update);

  RTCDevice rtc_device;
  bool rtc_device_is_sycl;
  enum RTCBuildQuality build_quality;
};

CCL_NAMESPACE_END

#endif /* WITH_EMBREE */
