/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Prism shell (research prototype, see PRIMITIVE_PRISM in kernel/types.h):
 * geometry and fog-style density math for a triangular prism formed by
 * extruding a mesh triangle's 3 vertices along their own PER-VERTEX
 * (smooth/shading) normals, each by a fixed thickness `h`, instead of
 * extruding the whole triangle along one shared face normal.
 *
 * Why per-vertex normals: two triangles that share an edge also share the
 * vertex normals at that edge's endpoints (unless the mesh has a hard/split
 * edge there, in which case their shading already disagrees, and the
 * shells correctly disagree too), so their extruded shells meet at exactly
 * the same 3D points along that edge -- no gap, no overlap.
 *
 * Still a 5-plane half-space clip, exactly like a single-shared-normal
 * prism -- NOT 8 discrete boundary triangles (an earlier version of this
 * code tried that, since a side "quad" between two independently-extruded
 * edges is generally non-planar; it worked but introduced its own
 * artificial seam at the arbitrary diagonal used to split each quad into 2
 * triangles, visible as banding even on a single isolated prism). The fix:
 * each of the 3 SIDE planes is swept along that EDGE's own averaged normal
 * -- normalize(N_i + N_j) for the edge between vertices i and j -- instead
 * of the single shared Nface. This keeps every plane exact and analytic
 * (one division each, no branching, no internal seam), and because vector
 * addition is symmetric, two adjacent triangles sharing an edge compute
 * bit-identical values for N_i + N_j (same two vertices' actual normals),
 * so their side planes for that shared edge match exactly -- genuine
 * edge-joining with none of the discrete-triangle-test machinery. The top
 * and bottom planes need no such trick: bottom is the real triangle's own
 * plane (unchanged), and top is the exact plane through the 3 individually
 * -extruded points (top_i = P_i + h*N_i) -- since adjacent triangles
 * already compute identical top_i values from the same shared vertex
 * normals, the outer edges naturally coincide too, with no extra
 * calculation needed. Only the side-plane sweep direction changes from the
 * single-normal version; "height above the surface" for the density
 * integral below still deliberately uses the triangle's ordinary FACE
 * normal (unchanged calculation), not any of the per-edge normals -- see
 * prism_density_integral for why that's kept simple and closed-form.
 */

#pragma once

#include "util/math_float3.h"

CCL_NAMESPACE_BEGIN

/* Prism shell self-intersection tag: a prism reuses its source triangle's
 * own global index (see BVHEmbree::add_prism_shell / add_object) rather
 * than having a separate index space, so the real surface and its prism
 * share the exact same `prim` value. After crossing a prism, the next
 * intersect_closest search's self-exclusion check needs to reject "the
 * same prism again" without ALSO rejecting "the real surface at that same
 * index" -- otherwise the real surface becomes permanently un-hittable
 * (confirmed: this caused the object to render fully see-through). ORing
 * this bit onto the persisted `isect.prim` when a prism hit continues
 * (kernel/integrator/shade_surface.h) tags it as "came from a prism", and
 * the prism intersect callback (kernel/device/cpu/bvh.h) compares against
 * the same tagged form -- a real triangle's untagged, small prim index can
 * never accidentally equal a tagged value, so it needs no changes at all
 * to stay hittable. 1<<28 leaves generous headroom below any realistic
 * mesh triangle count while staying far from PRIM_NONE (-1). */
#define PRISM_SELF_TAG (1 << 28)

/* Sentinel self-exclusion value (see PRISM_SELF_TAG above for the normal,
 * single-prism form) meaning "exclude every prism, unconditionally" for
 * the very next intersection search only -- written to isect.prim (see
 * shade_surface.h) once a ray has crossed more than PRISM_MAX_CROSSINGS in
 * a row without resolving to something real, as a way to guarantee that
 * search actually finds the real surface or background in one step instead
 * of potentially needing an unbounded number of further tiny crossings to
 * work through a dense overlap region. Distinct from PRISM_SELF_TAG (a
 * different bit) so the two can never be confused, and, like it, far above
 * any realistic mesh triangle count. Self-clearing: the next search's own
 * result overwrites isect.prim regardless of what it finds, so this only
 * ever affects that one search. */
#define PRISM_DISABLE_ALL_TAG (1 << 29)

/* Safety cap on how many prisms a single ray may cross in a row before it
 * gives up on prisms entirely for the rest of this burst (see
 * PRISM_DISABLE_ALL_TAG above and prism_crossings in
 * kernel/integrator/state_template.h). Not meant to ever be reached in
 * ordinary, lightly-overlapping scenes -- it's a backstop against dense
 * overlap regions where the per-crossing self-exclusion (a single
 * remembered prism) can make a ray revisit a small number of overlapping
 * prisms an astronomically large number of times before it would otherwise
 * clear the region on its own.
 *
 * Kept low (rather than the 256 used before edge-joined, per-vertex-normal
 * extrusion) specifically because edge-joining can make that backstop
 * trigger across an ENTIRE dense-overlap region on sharp-edged meshes, not
 * just a thin seam -- confirmed via a diagnostic marker lighting up the
 * whole overlap column between two overlapping cubes, where every burst
 * was running the full 256 crossings before giving up (each one now more
 * expensive too, from the extra per-crossing edge-normal math), making the
 * render dramatically slower. Since accumulated brightness is already
 * capped independently (see the surface_cap comment in shade_surface.h),
 * giving up sooner only costs a little accumulated color nuance in those
 * regions, not correctness -- so it's fine to bias hard toward speed here. */
#define PRISM_MAX_CROSSINGS 32

/* Which plane produced the exit (t_exit), for callers that need to know
 * whether it's safe to jump straight to a known neighboring triangle (see
 * PRISM_EXIT_EDGE0..2 and the fast-hop loop in shade_surface.h). */
#define PRISM_EXIT_BASE (0) /* the real mesh surface */
#define PRISM_EXIT_TOP (1) /* the shell's own outer boundary */
#define PRISM_EXIT_EDGE0 (2) /* side over edge v0-v1 */
#define PRISM_EXIT_EDGE1 (3) /* side over edge v1-v2 */
#define PRISM_EXIT_EDGE2 (4) /* side over edge v2-v0 */

/* Ray vs. the prism formed by extruding tri_P0/1/2 along their own
 * per-vertex normals tri_N0/1/2 by `thickness`, via 5-plane half-space
 * clipping -- see the file header comment for why each side plane sweeps
 * along that edge's own averaged normal rather than a single shared one,
 * and why that keeps this exact and seam-free like the original
 * single-normal version. Returns false if the ray misses the prism (or the
 * intersection interval falls entirely outside [tnear, tfar]). On success,
 * `Nface_out` is the triangle's own FACE normal (needed by the caller for
 * the height/emission integral, since it's already computed here), and
 * `exit_plane_out` is one of the PRISM_EXIT_* values above, identifying
 * which plane produced t_exit -- PRISM_EXIT_EDGE0/1/2 specifically means
 * "this boundary is exactly where a neighboring triangle's own prism would
 * begin, if that edge has one" (see tri_adjacency, kernel/data_arrays.h). */
ccl_device_inline bool ray_triangle_prism_intersect(const float3 ray_P,
                                                     const float3 ray_D,
                                                     const float tnear,
                                                     const float tfar,
                                                     const float3 tri_P0,
                                                     const float3 tri_P1,
                                                     const float3 tri_P2,
                                                     const float3 tri_N0,
                                                     const float3 tri_N1,
                                                     const float3 tri_N2,
                                                     const float thickness,
                                                     ccl_private float *t_enter,
                                                     ccl_private float *t_exit,
                                                     ccl_private float3 *Nface_out,
                                                     ccl_private int *exit_plane_out)
{
  const float3 Nface = normalize(cross(tri_P1 - tri_P0, tri_P2 - tri_P0));
  *Nface_out = Nface;

  const float3 top0 = tri_P0 + thickness * tri_N0;
  const float3 top1 = tri_P1 + thickness * tri_N1;
  const float3 top2 = tri_P2 + thickness * tri_N2;

  /* Edge-averaged sweep directions for the 3 side planes -- symmetric
   * (N_i + N_j order doesn't matter), so two triangles sharing an edge
   * compute the exact same value for it, and therefore the exact same side
   * plane. Falls back to Nface if a pair of vertex normals happen to
   * cancel out (near-180-degree opposed, e.g. a degenerate/inverted
   * normal) rather than dividing by a near-zero length. */
  const float3 sum01 = tri_N0 + tri_N1;
  const float3 sum12 = tri_N1 + tri_N2;
  const float3 sum20 = tri_N2 + tri_N0;
  const float3 edge_N01 = (dot(sum01, sum01) > 1e-12f) ? normalize(sum01) : Nface;
  const float3 edge_N12 = (dot(sum12, sum12) > 1e-12f) ? normalize(sum12) : Nface;
  const float3 edge_N20 = (dot(sum20, sum20) > 1e-12f) ? normalize(sum20) : Nface;

  float t_min = tnear;
  float t_max = tfar;
  int exit_plane = -1;

  /* Each entry: a point on the plane, and its OUTWARD normal (pointing away
   * from the prism's interior). Bottom is the triangle's own plane
   * (unchanged); top is the exact plane through the 3 individually
   * extruded points (not necessarily parallel to the bottom anymore); the
   * 3 sides sweep each edge along that edge's own averaged normal (see
   * above), not a single shared direction. */
  const float3 plane_point[5] = {
      tri_P0,
      top0,
      tri_P0,
      tri_P1,
      tri_P2,
  };
  /* All 5 normalized to unit length -- the 3 side ones (cross of an EDGE
   * VECTOR, not a unit vector, with a unit edge normal) would otherwise
   * come out scaled by that edge's own length, making the near-zero-
   * denominator check below (which compares against a fixed absolute
   * threshold) meaningless for them: a ray genuinely near-tangent to a
   * side plane could still produce a "not quite small enough" denom that
   * slips past the check, then blows up into a wildly incorrect t from
   * dist/denom instead of being safely treated as non-constraining. With
   * every plane_normal at unit length, denom is a true direction cosine
   * for all 5, so one consistent threshold means the same thing (a
   * genuine near-parallel angular tolerance) everywhere. */
  float3 plane_normal[5] = {
      -Nface,
      normalize(cross(top1 - top0, top2 - top0)),
      normalize(cross(tri_P1 - tri_P0, edge_N01)),
      normalize(cross(tri_P2 - tri_P1, edge_N12)),
      normalize(cross(tri_P0 - tri_P2, edge_N20)),
  };
  /* Top and side normals need to point away from the prism's interior to be
   * truly outward -- fix the sign using the triangle's own centroid, which
   * is guaranteed to be on the inward side of every one of these planes. */
  const float3 centroid = (tri_P0 + tri_P1 + tri_P2) * (1.0f / 3.0f);
  for (int i = 1; i < 5; i++) {
    if (dot(centroid - plane_point[i], plane_normal[i]) > 0.0f) {
      plane_normal[i] = -plane_normal[i];
    }
  }

  for (int i = 0; i < 5; i++) {
    const float denom = dot(ray_D, plane_normal[i]);
    const float dist = dot(plane_point[i] - ray_P, plane_normal[i]);

    if (fabsf(denom) < 1e-5f) {
      /* Ray parallel to this plane: either entirely inside or entirely
       * outside the half-space, for the whole ray. Now that every
       * plane_normal is unit length, denom is a true direction cosine, so
       * this threshold is a genuine (if still tight) angular tolerance
       * rather than an edge-length-dependent one -- widened from the
       * original 1e-8f (which, even as a cosine, is tighter than single-
       * precision round-off in the dot product itself can reliably
       * resolve) to actually catch near-tangent cases instead of falling
       * through to an unstable dist/denom division. */
      if (dist < 0.0f) {
        /* ray_P is on the outward side already -- no intersection. */
        return false;
      }
      continue;
    }

    const float t = dist / denom;
    if (denom < 0.0f) {
      /* Ray heading into the half-space: this is a potential entry. */
      t_min = max(t_min, t);
    }
    else {
      /* Ray heading out of the half-space: this is a potential exit. */
      if (t < t_max) {
        t_max = t;
        exit_plane = i;
      }
    }
  }

  if (t_min > t_max) {
    return false;
  }

  *t_enter = t_min;
  *t_exit = t_max;
  /* exit_plane stays -1 (never updated) only if no plane's exit-t ever beat
   * the initial tfar -- i.e. the ray's own tfar bound is what limited it,
   * not any of the prism's own boundaries. Treat that as PRISM_EXIT_TOP
   * (not fast-hoppable): there's no real surface there either, so this
   * only matters for whether the caller tries to fast-hop, and "no" is
   * always safe. */
  *exit_plane_out = (exit_plane >= 0) ? exit_plane : PRISM_EXIT_TOP;
  return true;
}

/* Closed-form integral of density(height) = clamp(1 - height/radius, 0,
 * 1)^power along the ray segment [t_enter, t_exit], where height(t) =
 * dot(ray_P + t*ray_D - tri_P0, Nface) varies linearly in t (see this
 * file's header comment for why that's only exact for a face-normal, not
 * vertex-normal, extrusion). This is what makes a prism crossing behave
 * like passing through a genuine fog/volume slab rather than sampling a
 * single point on a thin shell: the contribution scales with how much of
 * the density field the ray actually traveled through, not just where it
 * entered.
 *
 * A clamped power curve rather than an unbounded exponential/inverse-square
 * tail deliberately: density must reach EXACTLY 0 at height = radius (the
 * shell's own geometric boundary), not just asymptotically approach it --
 * otherwise a ray exiting the shell drops a still-visible glow value to
 * nothing in a single step, showing up as a hard edge right at the outer
 * boundary instead of a soft fade. `power` (kobject.prism_falloff_power,
 * user-controlled) shapes how front-loaded that falloff is: 1.0 is a plain
 * linear ramp, higher values concentrate brightness closer to the surface
 * before dropping to the edge, lower values (> 0) stay brighter for longer
 * first. Any power > 1 also makes the fade C1-continuous (zero slope) at
 * height = radius, avoiding a visible kink there.
 *
 * Substituting u(t) = 1 - height(t)/radius (linear in t, u = 1 at the
 * surface, u = 0 at the boundary) turns this into ∫ clamp(u, 0, 1)^power
 * dt, a piecewise integral: a flat "u >= 1" region (density pinned at 1,
 * for height <= 0, e.g. very close to a concave edge where a neighboring
 * prism's own base dips below this one), a polynomial ramp through the
 * region where u is actually between 0 and 1 (closed form via the power
 * rule for ANY real power > 0, not just integers: the antiderivative of
 * u(t)^power = (C-Dt)^power is -(C-Dt)^(power+1) / ((power+1)*D)), and a
 * flat "u <= 0" region (density 0, height > radius) -- each contributing
 * only the sub-length of [t_enter, t_exit] that actually falls in it.
 * Falls back to the midpoint density (numerically stable) when the ray
 * runs nearly parallel to the triangle's plane, where u(t) is ~constant
 * and the ramp's closed form has a removable 0/0 singularity. */
ccl_device_inline float prism_density_integral(const float3 ray_P,
                                                const float3 ray_D,
                                                const float3 tri_P0,
                                                const float3 Nface,
                                                const float radius,
                                                const float power,
                                                const float t_enter,
                                                const float t_exit)
{
  const float A = dot(ray_P - tri_P0, Nface);
  const float B = dot(ray_D, Nface);

  /* u(t) = C - D*t. */
  const float C = 1.0f - A / radius;
  const float D = B / radius;

  if (fabsf(D) < 1e-6f) {
    const float u = clamp(C - D * (0.5f * (t_enter + t_exit)), 0.0f, 1.0f);
    return (t_exit - t_enter) * powf(u, power);
  }

  /* t where u crosses 1 (height = 0) and 0 (height = radius). Whichever of
   * these is smaller marks the start of the polynomial ramp; whichever is
   * larger marks its end. Which one is which depends on the sign of D. */
  const float t_at_u1 = (C - 1.0f) / D;
  const float t_at_u0 = C / D;
  const float t_ramp_lo = min(t_at_u1, t_at_u0);
  const float t_ramp_hi = max(t_at_u1, t_at_u0);
  /* D > 0 means u decreases as t increases, so the "u >= 1" flat region
   * (if any) comes BEFORE the ramp, and "u <= 0" comes after; D < 0 is the
   * mirror image. */
  const bool high_plateau_is_before_ramp = (D > 0.0f);

  float total = 0.0f;

  const float pre_ramp_end = min(t_exit, t_ramp_lo);
  if (high_plateau_is_before_ramp && pre_ramp_end > t_enter) {
    total += pre_ramp_end - t_enter;
  }

  const float ramp_lo = max(t_enter, t_ramp_lo);
  const float ramp_hi = min(t_exit, t_ramp_hi);
  if (ramp_hi > ramp_lo) {
    const float u_lo = C - D * ramp_lo;
    const float u_hi = C - D * ramp_hi;
    const float power1 = power + 1.0f;
    total += (powf(u_lo, power1) - powf(u_hi, power1)) / (power1 * D);
  }

  const float post_ramp_start = max(t_enter, t_ramp_hi);
  if (!high_plateau_is_before_ramp && t_exit > post_ramp_start) {
    total += t_exit - post_ramp_start;
  }

  return total;
}

CCL_NAMESPACE_END
