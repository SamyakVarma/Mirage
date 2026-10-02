/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Joy skin sparkle field (research prototype): a sparse scatter of small
 * glow points near a surface, thinning out with distance -- additive on top
 * of the prism shell's own continuous glow (see shade_surface.h). Validated
 * first as a standalone Python/PyVista prototype (spatial distribution only,
 * no lighting) before writing this kernel version.
 *
 * Each 3D grid cell independently gets, from a pure hash of its integer
 * coordinates (util/hash.h's existing PCG-based hash, the same primitive the
 * built-in Voronoi texture node uses -- no stored hashmap, no precompute,
 * unbounded resolution): a jittered center point inside the cell, a random
 * base radius, and a random "existence roll". A cell's sphere only exists if
 * its roll beats a keep_probability that depends on distance-to-surface (~1
 * near the surface, ~0 past max_dist); radius also shrinks with distance.
 * Near the surface this produces many overlapping big spheres that read as
 * a continuous glow; far away, few small isolated ones that read as sparse
 * dust -- without needing cell size itself to vary. */

#pragma once

#include "util/hash.h"
#include "util/math_float3.h"

CCL_NAMESPACE_BEGIN

/* Genuinely volumetric, not surface-bound: candidate sparkle spheres are
 * tested against the ray's ACTUAL closest approach anywhere within
 * [t_enter, t_exit] (a proper analytic point-to-segment distance, clamped
 * into range), not against one fixed sample point -- so a sphere sitting
 * deep near t_enter, right at t_exit, or anywhere between is detected the
 * same way, giving the field real depth through the shell's thickness
 * rather than collapsing onto wherever a single sample happened to land.
 * The only thing sampled along the segment is which GRID CELLS to examine
 * in the first place (stepped at roughly one sample per cell_size of
 * travel, capped for cost) -- a long crossing spans multiple cells along
 * the ray's direction, and skipping that would miss spheres in cells a
 * single sample point never touches, which is exactly the bug this
 * replaces (evaluating only at the segment's midpoint meant spheres away
 * from that one depth were silently invisible). tri_P0/Nface are the SAME
 * per-triangle plane already used for the aura's own height/falloff (see
 * prism_density_integral in math_prism.h): distance-to-surface for each
 * CANDIDATE SPHERE's own center is approximated the same flat-plane way,
 * for consistency with the shell's own approximation. Returns a plain
 * [0,1]-ish weight; the caller multiplies this by its own strength/hue,
 * exactly like the aura's own density term. */
ccl_device_inline float sparkle_field_eval(const float3 ray_P,
                                            const float3 ray_D,
                                            const float t_enter,
                                            const float t_exit,
                                            const float3 tri_P0,
                                            const float3 Nface,
                                            const float cell_size,
                                            const float max_dist,
                                            const float radius_near,
                                            const float radius_far,
                                            const float keep_near,
                                            const float keep_far,
                                            const float falloff_power,
                                            ccl_private float3 *out_center = nullptr)
{
  if (cell_size <= 0.0f || max_dist <= 0.0f) {
    return 0.0f;
  }

  const float seg_len = t_exit - t_enter;
  const int num_steps = clamp(int(ceilf(seg_len / max(cell_size, 1e-6f))), 1, 8);

  float best_coverage = 0.0f; /* 1 - dist/radius of the strongest hit so far. */
  float3 best_center = zero_float3();
  bool found = false;

  for (int step = 0; step < num_steps; step++) {
    const float t_step = t_enter + (float(step) + 0.5f) * (seg_len / float(num_steps));
    const float3 step_P = ray_P + t_step * ray_D;
    const float3 cell_f = floor(step_P / cell_size);
    const int3 cell = make_int3(cell_f);

    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          const int3 neighbor = cell + make_int3(dx, dy, dz);

          /* Two independent hash calls (salt 0 and 1 in the 4th component)
           * to get 5 pseudo-random channels per cell: jitter offset (xyz),
           * a radius roll, and an existence roll. */
          const float4 h0 = hash_int4_to_float4(
              make_int4(neighbor.x, neighbor.y, neighbor.z, 0));
          const float exist_r =
              hash_int4_to_float4(make_int4(neighbor.x, neighbor.y, neighbor.z, 1)).x;

          const float3 center = (make_float3(neighbor) + make_float3(h0.x, h0.y, h0.z)) *
                                cell_size;

          /* Distance from THIS candidate sphere's own center to the surface. */
          const float height = dot(center - tri_P0, Nface);
          const float t_norm = clamp(height, 0.0f, max_dist) / max_dist;
          const float w = powf(clamp(1.0f - t_norm, 0.0f, 1.0f), falloff_power);

          const float keep_probability = keep_far + (keep_near - keep_far) * w;
          if (exist_r >= keep_probability) {
            continue; /* This cell's sphere doesn't exist. */
          }

          /* Analytic closest approach of the ACTUAL ray segment (not just
           * this lookup's step point) to this candidate sphere's center --
           * this is what gives the field real depth through the shell. */
          const float t_closest = clamp(dot(center - ray_P, ray_D), t_enter, t_exit);
          const float dist = len(ray_P + t_closest * ray_D - center);

          const float radius = radius_far + (radius_near - radius_far) * w;
          if (dist < radius) {
            const float coverage = 1.0f - dist / radius;
            if (coverage > best_coverage) {
              best_coverage = coverage;
              best_center = center;
              found = true;
            }
          }
        }
      }
    }
  }

  if (!found) {
    return 0.0f;
  }

  if (out_center != nullptr) {
    *out_center = best_center;
  }

  return powf(clamp(best_coverage, 0.0f, 1.0f), falloff_power);
}

CCL_NAMESPACE_END
