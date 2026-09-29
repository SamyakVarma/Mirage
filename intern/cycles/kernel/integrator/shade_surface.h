/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/path_state.h"
#include "kernel/integrator/surface_shader.h"

#include "kernel/film/data_passes.h"
#include "kernel/film/denoising_passes.h"
#include "kernel/film/light_passes.h"

#include "kernel/light/sample.h"

#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/shadow_linking.h"
#include "kernel/integrator/subsurface.h"
#include "kernel/integrator/volume_stack.h"

#include "kernel/types.h"
#include "kernel/util/colorspace.h"
#include "kernel/util/math_prism.h"
#include "util/math_intersect.h"

CCL_NAMESPACE_BEGIN

ccl_device_forceinline void integrate_surface_shader_setup(KernelGlobals kg,
                                                           ConstIntegratorState state,
                                                           ccl_private ShaderData *sd)
{
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(state, &isect);

  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &ray);

  shader_setup_from_ray(kg, sd, &ray, &isect);

#ifdef __SPECTRAL__
  shader_setup_wavelength(kg, sd, state);
#endif
}

ccl_device_forceinline float3 integrate_surface_ray_offset(KernelGlobals kg,
                                                           const ccl_private ShaderData *sd,
                                                           const float3 ray_P,
                                                           const float3 ray_D)
{
  /* No ray offset needed for other primitive types. */
  if (!(sd->type & PRIMITIVE_TRIANGLE)) {
    return ray_P;
  }

  /* Self intersection tests already account for the case where a ray hits the
   * same primitive. However precision issues can still cause neighboring
   * triangles to be hit. Here we test if the ray-triangle intersection with
   * the same primitive would miss, implying that a neighboring triangle would
   * be hit instead.
   *
   * This relies on triangle intersection to be watertight, and the object inverse
   * object transform to match the one used by ray intersection exactly.
   *
   * Potential improvements:
   * - It appears this happens when either barycentric coordinates are small,
   *   or dot(sd->Ng, ray_D)  is small. Detect such cases and skip test?
   * - Instead of ray offset, can we tweak P to lie within the triangle?
   */

  /* TODO: Investigate if there are better ray offsetting algorithms for each BVH.
   * Cycles and Custom BVH triangle tests aren't numerically identical, meaning
   * this method isn't ideal for them. */

  float3 verts[3];
  if (sd->type == PRIMITIVE_TRIANGLE) {
    triangle_vertices(kg, sd->object, sd->prim, verts);
  }
  else {
    kernel_assert(sd->type == PRIMITIVE_MOTION_TRIANGLE);
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, verts);
  }

  float3 local_ray_P = ray_P;
  float3 local_ray_D = ray_D;

  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, sd);
    local_ray_P = transform_point(&itfm, local_ray_P);
    local_ray_D = transform_direction(&itfm, local_ray_D);
  }

  if (ray_triangle_intersect_self(local_ray_P, local_ray_D, verts)) {
    return ray_P;
  }
  return ray_offset(ray_P, sd->Ng);
}

ccl_device_forceinline bool integrate_surface_holdout(KernelGlobals kg,
                                                      ConstIntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      ccl_global float *ccl_restrict render_buffer)
{
  /* Write holdout transparency to render buffer and stop if fully holdout. */
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (((sd->runtime_flag & SR_HOLDOUT) || (sd->object_flag & SD_OBJECT_HOLDOUT_MASK)) &&
      (path_flag & PATH_RAY_TRANSPARENT_BACKGROUND))
  {
    const Spectrum holdout_weight = surface_shader_apply_holdout(sd);
    const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
    const float transparent = average(holdout_weight * throughput);
    film_write_holdout(kg, state, path_flag, transparent, render_buffer);
    if (isequal(holdout_weight, one_spectrum())) {
      return false;
    }
  }

  return true;
}

ccl_device_forceinline void integrate_surface_emission(KernelGlobals kg,
                                                       IntegratorState state,
                                                       const ccl_private ShaderData *sd,
                                                       ccl_global float *ccl_restrict
                                                           render_buffer)
{
  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

#ifdef __LIGHT_LINKING__
  if (!(path_visibility & PATH_RAY_VISIBILITY_CAMERA) &&
      !light_link_object_match(kg, light_link_receiver_forward(kg, state), sd->object))
  {
    return;
  }
#endif

#ifdef __SHADOW_LINKING__
  /* Indirect emission of shadow-linked emissive surfaces is done via shadow rays to dedicated
   * light sources. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_LINKING) {
    if (!(path_visibility & PATH_RAY_VISIBILITY_CAMERA) &&
        kernel_data_fetch(objects, sd->object).shadow_set_membership != LIGHT_LINK_MASK_ALL)
    {
      return;
    }
  }
#endif

  /* Evaluate emissive closure. */
  const Spectrum L = surface_shader_emission(sd);

  const float mis_weight = light_sample_mis_weight_forward_surface(
      kg, state, path_visibility, path_flag, sd);

  guiding_record_surface_emission(kg, state, L, mis_weight);
  film_write_surface_emission(
      kg, state, L, mis_weight, render_buffer, object_lightgroup(kg, sd->object));
}

ccl_device int integrate_surface_ray_portal(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            const ccl_private ShaderClosure *sc)
{
  const ccl_private RayPortalClosure *pc = (const ccl_private RayPortalClosure *)sc;

  float sum_sample_weight = 0.0f;
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      sum_sample_weight += sc->sample_weight;
    }
  }
  if (sum_sample_weight <= 0.0f) {
    return LABEL_NONE;
  }

  if (len_squared(sd->P - pc->P) > 1e-9f) {
    /* if the ray origin is changed, unset the current object,
     * so we can potentially hit the same polygon again */
    INTEGRATOR_STATE_WRITE(state, isect, object) = OBJECT_NONE;
    INTEGRATOR_STATE_WRITE(state, ray, P) = pc->P;
  }
  else {
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, pc->P, pc->D);
  }
  INTEGRATOR_STATE_WRITE(state, ray, D) = pc->D;
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
  INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
  INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);
#endif

  const float pick_pdf = pc->sample_weight / sum_sample_weight;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= pc->weight / pick_pdf;

  const int label = LABEL_TRANSMIT | LABEL_RAY_PORTAL;
  path_state_next(kg, state, label, sd->runtime_flag);

  return label;
}

/* Branch off a shadow path and initialize common part of it.
 * THe common is between the surface shading and configuration of a special shadow ray for the
 * shadow linking. */
ccl_device_inline IntegratorShadowState
integrate_direct_light_shadow_init_common(KernelGlobals kg,
                                          IntegratorState state,
                                          const ccl_private Ray *ccl_restrict ray,
                                          const Spectrum bsdf_spectrum,
                                          const int light_group,
                                          const int mnee_vertex_count,
                                          const bool constant_light_shader)
{
  const DeviceKernel next_kernel = (constant_light_shader) ?
                                       DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW :
                                       DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE;

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state;
#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    /* Reuse shadow path that was already allocated by intersect_mnee. */
    shadow_state = integrator_state_get_mnee_shadow_state(state);
    integrator_shadow_path_next(
        shadow_state, DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING, next_kernel);
  }
  else
#endif
  {
    shadow_state = integrator_shadow_path_init(kg, state, next_kernel, false);
  }

#ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, ray);
  integrator_state_write_shadow_ray_self(shadow_state, ray);

  /* Copy state from main path to shadow path. */
  const Spectrum unlit_throughput = INTEGRATOR_STATE(state, path, throughput);
  const Spectrum throughput = unlit_throughput * bsdf_spectrum;

  if (!(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bsdf_eval_average) = average(bsdf_spectrum);
  }

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = INTEGRATOR_STATE(
      state, path, transparent_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, glossy_bounce) = INTEGRATOR_STATE(
      state, path, glossy_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if ((kernel_data.kernel_features & KERNEL_FEATURE_NODE_PORTAL)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, portal_bounce) = INTEGRATOR_STATE(
        state, path, portal_bounce);
  }

#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) =
        INTEGRATOR_STATE(state, path, transmission_bounce) + mnee_vertex_count - 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           diffuse_bounce) = INTEGRATOR_STATE(state, path, diffuse_bounce) + 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           bounce) = INTEGRATOR_STATE(state, path, bounce) + mnee_vertex_count;
  }
  else
#endif
  {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) = INTEGRATOR_STATE(
        state, path, transmission_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, diffuse_bounce) = INTEGRATOR_STATE(
        state, path, diffuse_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = INTEGRATOR_STATE(
        state, path, bounce);
  }

  /* Write Light-group, +1 as light-group is int but we need to encode into a uint8_t. */
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = light_group + 1;

#if defined(__PATH_GUIDING__)
  if ((kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unlit_throughput) = unlit_throughput;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, path_segment) = INTEGRATOR_STATE(
        state, guiding, path_segment);
    INTEGRATOR_STATE(shadow_state, shadow_path, guiding_light_linking_mis_weight) = 0.0f;
  }
#endif

  return shadow_state;
}

/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
template<uint64_t node_feature_mask>
#if defined(__KERNEL_HIP__)
/* Inlining the function makes gfx1102 crash rendering principled_bsdf_bevel_emission_137420.blend
 * using SDK 7.2.1. */
ccl_device_noinline
#elif defined(__KERNEL_GPU__)
ccl_device_forceinline
#else
/* MSVC has very long compilation time (x20) if we force inline this function */
ccl_device
#endif
    ShaderEvalResult
    integrate_surface_direct_light(KernelGlobals kg,
                                   IntegratorState state,
                                   ccl_private ShaderData *sd,
                                   const ccl_private RNGState *rng_state)
{
  /* Test if there is a light or BSDF that needs direct light. */
  if (!(kernel_data.integrator.use_direct_light && (sd->runtime_flag & SR_BSDF_HAS_EVAL))) {
    return SHADER_EVAL_EMPTY;
  }

  LightSample ls ccl_optional_struct_init;
  int mnee_vertex_count = 0;  // NOLINT

#ifdef __MNEE__
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      (INTEGRATOR_STATE(state, path, mnee) & PATH_MNEE_SAMPLED))
  {
    /* MNEE already sampled a light and caustics casters. */
    integrator_state_read_mnee(state, &ls, &mnee_vertex_count);
  }
  else
#endif
  {
    /* Sample position on a light. */
    const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
    const uint bounce = INTEGRATOR_STATE(state, path, bounce);
    const float3 rand_light = path_state_rng_3D(kg, rng_state, PRNG_LIGHT);

    if (!light_sample_from_position(kg,
                                    rand_light,
                                    sd->time,
                                    sd->P,
                                    sd->N,
                                    light_link_receiver_nee(kg, sd),
                                    sd->runtime_flag,
                                    bounce,
                                    path_flag,
                                    &ls))
    {
      return SHADER_EVAL_EMPTY;
    }
  }

  kernel_assert(ls.pdf != 0.0f);

  const bool is_transmission = dot(ls.D, sd->N) < 0.0f;

  if (ls.prim != PRIM_NONE && ls.prim == sd->prim && ls.object == sd->object) {
    /* Skip self intersection if light direction lies in the same hemisphere as the geometric
     * normal. */
    if (dot(ls.D, is_transmission ? -sd->Ng : sd->Ng) > 0.0f) {
      return SHADER_EVAL_EMPTY;
    }
  }

#ifdef __MNEE__
  /* On a caustic caster, a caustic light's contribution is delivered to receivers by
   * MNEE and does not need to be computed again here. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_MNEE) {
    if (mnee_vertex_count == 0 && is_transmission &&
        (sd->object_flag & SD_OBJECT_CAUSTICS_CASTER) && ls.type != LIGHT_TRIANGLE &&
        kernel_data_fetch(lights, ls.prim).use_caustics)
    {
      return SHADER_EVAL_EMPTY;
    }
  }
#endif

  /* Evaluate constant part of light shader, rest will optionally be done in another kernel. */
  Spectrum light_shader_eval ccl_optional_struct_init;
  const bool is_constant_light_shader = light_sample_shader_eval_nee_constant(
      kg, ls.shader, ls.prim, ls.type != LIGHT_TRIANGLE, light_shader_eval);

  /* Evaluate BSDF. */
  BsdfEval bsdf_eval ccl_optional_struct_init;
  float avg_roughness_squared = 0.0f;
  const float bsdf_pdf = surface_shader_bsdf_eval(
      kg, state, sd, ls.D, &bsdf_eval, ls.shader, avg_roughness_squared);

  Ray ray ccl_optional_struct_init;

#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    light_shader_eval *= integrator_state_read_mnee_throughput(state);
    bsdf_eval_mul(&bsdf_eval, light_shader_eval);

    if (bsdf_eval_is_zero(&bsdf_eval)) {
      return SHADER_EVAL_EMPTY;
    }

    integrator_state_read_mnee_ray(state, &ls, &ray);
  }
  else
#endif /* __MNEE__ */
  {
    const float mis_weight = light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
    bsdf_eval_mul(&bsdf_eval, light_shader_eval * ls.eval_fac / ls.pdf * mis_weight);

    /* Path termination for constant light shader. */
    if (is_constant_light_shader && !(kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_TREE)) {
      const float terminate = path_state_rng_light_termination(kg, rng_state);
      if (light_sample_terminate(kg, &bsdf_eval, terminate)) {
        return SHADER_EVAL_EMPTY;
      }
    }
    /* For non-constant light shader, probabilistic termination happens in
     * SHADE_LIGHT_NEE when the full contribution is known. */
    else if (bsdf_eval_is_zero(&bsdf_eval)) {
      return SHADER_EVAL_EMPTY;
    }

    /* Create shadow ray. */
    light_sample_to_surface_shadow_ray(kg, sd, &ls, &ray);

#ifdef __RAY_DIFFERENTIALS__
    /* Widen ray differences, with same logic as forward sampling to ensure
     * both MIS strategies converge to the same result. */
    ray.dD = bsdf_widen_dD(kg, INTEGRATOR_STATE(state, ray, dD), avg_roughness_squared);
#endif
  }

  if (ray.self.object != OBJECT_NONE) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg,
      state,
      &ray,
      bsdf_eval_sum(&bsdf_eval),
      ls.group,
      mnee_vertex_count,
      is_constant_light_shader);

  if (is_transmission) {
#ifdef __VOLUME__
    volume_stack_enter_exit<true>(kg, shadow_state, sd);
#endif
  }

  uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag);

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    PackedSpectrum pass_diffuse_weight;
    PackedSpectrum pass_glossy_weight;

    if (shadow_flag & PATH_RAY_ANY_PASS) {
      /* Indirect bounce, use weights from earlier surface or volume bounce. */
      pass_diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
      pass_glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
    }
    else {
      /* Direct light, use BSDFs at this bounce. */
      shadow_flag |= PATH_RAY_SURFACE_PASS;
      pass_diffuse_weight = PackedSpectrum(bsdf_eval_pass_diffuse_weight(&bsdf_eval));
      pass_glossy_weight = PackedSpectrum(bsdf_eval_pass_glossy_weight(&bsdf_eval));
    }

    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = pass_diffuse_weight;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = pass_glossy_weight;
  }

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = INTEGRATOR_STATE(
      state, path, visibility);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;

  return SHADER_EVAL_OK;
}

/* Path tracing: bounce off or through surface with new direction. */
ccl_device_forceinline int integrate_surface_bsdf_bssrdf_bounce(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private RNGState *rng_state)
{
  /* Sample BSDF or BSSRDF. */
  if (!(sd->runtime_flag & (SR_BSDF | SR_BSSRDF))) {
    return LABEL_NONE;
  }

  float3 rand_bsdf = path_state_rng_3D(kg, rng_state, PRNG_SURFACE_BSDF);
  const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(sd, &rand_bsdf);

#ifdef __SUBSURFACE__
  /* BSSRDF closure, we schedule subsurface intersection kernel. */
  if (CLOSURE_IS_BSSRDF(sc->type)) {
    return subsurface_bounce(kg, state, sd, sc);
  }
#endif
  if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
    return integrate_surface_ray_portal(kg, state, sd, sc);
  }

  /* BSDF closure, sample direction. */
  float bsdf_pdf = 0.0f;
  float unguided_bsdf_pdf = 0.0f;
  BsdfEval bsdf_eval ccl_optional_struct_init;
  float3 bsdf_wo ccl_optional_struct_init;
  int label;

  float2 bsdf_sampled_roughness = make_float2(1.0f, 1.0f);
  float bsdf_eta = 1.0f;
  float mis_pdf = 1.0f;

  float bsdf_avg_roughness_squared = 0.0f;

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
  if (kernel_data.integrator.use_surface_guiding &&
      (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING))
  {
    label = surface_shader_bsdf_guided_sample_closure(kg,
                                                      state,
                                                      sd,
                                                      sc,
                                                      rand_bsdf,
                                                      &bsdf_eval,
                                                      &bsdf_wo,
                                                      &bsdf_pdf,
                                                      &mis_pdf,
                                                      &unguided_bsdf_pdf,
                                                      &bsdf_sampled_roughness,
                                                      &bsdf_eta,
                                                      rng_state,
                                                      bsdf_avg_roughness_squared);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }

    INTEGRATOR_STATE_WRITE(state, path, unguided_throughput) *= bsdf_pdf / unguided_bsdf_pdf;
  }
  else
#endif
  {
    label = surface_shader_bsdf_sample_closure(kg,
                                               sd,
                                               sc,
                                               rand_bsdf,
                                               &bsdf_eval,
                                               &bsdf_wo,
                                               &bsdf_pdf,
                                               &bsdf_sampled_roughness,
                                               &bsdf_eta,
                                               bsdf_avg_roughness_squared);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }
    mis_pdf = bsdf_pdf;
    unguided_bsdf_pdf = bsdf_pdf;
  }

  if (label & LABEL_TRANSPARENT) {
    /* Only need to modify start distance for transparent. */
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);
  }
  else {
    /* Setup ray with changed origin and direction. */
    const float3 D = normalize(bsdf_wo);
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, sd->P, D);
    INTEGRATOR_STATE_WRITE(state, ray, D) = D;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
    INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);

    /* Widen ray differences, with same logic as NEE sampling to ensure
     * both MIS strategies converge to the same result. */
    const float dD = bsdf_widen_dD(
        kg, INTEGRATOR_STATE(state, ray, dD), bsdf_avg_roughness_squared);
    INTEGRATOR_STATE_WRITE(state, ray, dD) = dD;
#endif
  }

  /* Update throughput. */
  const Spectrum bsdf_weight = bsdf_eval_sum(&bsdf_eval) / bsdf_pdf;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= bsdf_weight;

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    if (INTEGRATOR_STATE(state, path, bounce) == 0) {
      INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) = bsdf_eval_pass_diffuse_weight(
          &bsdf_eval);
      INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) = bsdf_eval_pass_glossy_weight(
          &bsdf_eval);
    }
  }

  /* Update path state */
  if (!(label & LABEL_TRANSPARENT)) {
    const float min_ray_pdf = INTEGRATOR_STATE(state, path, min_ray_pdf);
    INTEGRATOR_STATE_WRITE(state, path, mis_ray_pdf) = mis_pdf;
    INTEGRATOR_STATE_WRITE(state, path, mis_origin_n) = sd->N;
    INTEGRATOR_STATE_WRITE(state, path, min_ray_pdf) = fminf(unguided_bsdf_pdf, min_ray_pdf);

#ifdef __LIGHT_LINKING__
    if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) {
      INTEGRATOR_STATE_WRITE(state, path, mis_ray_object) = sd->object;
    }
#endif
  }

  path_state_next(kg, state, label, sd->runtime_flag);

  guiding_record_surface_bounce(kg,
                                state,
                                bsdf_weight,
                                bsdf_pdf,
                                sd->N,
                                normalize(bsdf_wo),
                                bsdf_sampled_roughness,
                                bsdf_eta);

  return label;
}

#ifdef __VOLUME__
ccl_device_forceinline int integrate_surface_volume_only_bounce(IntegratorState state,
                                                                ccl_private ShaderData *sd)
{
  if (!path_state_volume_next(state)) {
    return LABEL_NONE;
  }

  /* Only modify start distance. */
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);

  return LABEL_TRANSMIT | LABEL_TRANSPARENT;
}
#endif

ccl_device_forceinline bool integrate_surface_terminate(IntegratorState state,
                                                        const uint32_t path_flag)
{
  const float continuation_probability = (path_flag & PATH_RAY_TERMINATE_ON_NEXT_SURFACE) ?
                                             0.0f :
                                             float(INTEGRATOR_STATE(
                                                 state, path, continuation_probability));
  if (continuation_probability == 0.0f) {
    return true;
  }
  if (continuation_probability != 1.0f) {
    INTEGRATOR_STATE_WRITE(state, path, throughput) /= continuation_probability;
  }

  return false;
}

#if defined(__AO__)
ccl_device_forceinline void integrate_surface_ao(KernelGlobals kg,
                                                 IntegratorState state,
                                                 const ccl_private ShaderData *ccl_restrict sd,
                                                 const ccl_private RNGState *ccl_restrict
                                                     rng_state)
{
  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (!(kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) &&
      !(path_visibility & PATH_RAY_VISIBILITY_CAMERA))
  {
    return;
  }

  /* Skip AO for paths that were split off for shadow catchers to avoid double-counting. */
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  const float2 rand_bsdf = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);

  float3 ao_N;
  const Spectrum ao_weight = surface_shader_ao(
      sd, kernel_data.integrator.ao_additive_factor, &ao_N);

  float3 ao_D;
  float ao_pdf;
  sample_cos_hemisphere(ao_N, rand_bsdf, &ao_D, &ao_pdf);

  bool skip_self = true;

  Ray ray ccl_optional_struct_init;
  ray.P = shadow_ray_offset(kg, sd, ao_D, &skip_self);
  ray.D = ao_D;
  if (skip_self) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }
  ray.tmin = 0.0f;
  ray.tmax = kernel_data.integrator.ao_bounces_distance;
  ray.time = sd->time;
  ray.self.object = (skip_self) ? sd->object : OBJECT_NONE;
  ray.self.prim = (skip_self) ? sd->prim : PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, true);

#  ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#  endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, &ray);
  integrator_state_write_shadow_ray_self(shadow_state, &ray);

  /* Copy state from main path to shadow path. */
  const uint16_t bounce = INTEGRATOR_STATE(state, path, bounce);
  const uint16_t transparent_bounce = INTEGRATOR_STATE(state, path, transparent_bounce);
  const uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag) | PATH_RAY_SHADOW_FOR_AO;
  const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput) * surface_shader_alpha(sd);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, visibility) = INTEGRATOR_STATE(
      state, path, visibility);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = transparent_bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if (kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unshadowed_throughput) = ao_weight;
  }
}
#endif /* defined(__AO__) */

template<uint64_t node_feature_mask>
ccl_device int integrate_surface(KernelGlobals kg,
                                 IntegratorState state,
                                 ccl_global float *ccl_restrict render_buffer)

{
  PROFILING_INIT_FOR_SHADER(kg, PROFILING_SHADE_SURFACE_SETUP);

  /* Setup shader data. */
  ShaderData sd;
  integrate_surface_shader_setup(kg, state, &sd);
  PROFILING_SHADER(sd.object, sd.shader);

  /* Prism shell (research prototype): fog/volume-style -- a hit is the ray
   * entering a slab of density, NOT a real surface (no BSDF, no shadow rays
   * of its own, no bounce). The ray keeps going afterward, and may cross
   * more prisms before finally resolving to a real surface or the
   * background -- see the flush points below (real surface) and in
   * integrate_background(). Each crossing's contribution is added on top
   * of what's already accumulated, but the running total is capped at that
   * crossing's own "at the surface" brightness (see the cap comment
   * further down) -- so overlapping shells still blend and brighten
   * locally like real overlapping fog, without ever summing past a sane
   * ceiling regardless of how many prisms a ray ends up crossing.
   *
   * Crossing a prism whose exit is through an edge SHARED with a known
   * neighboring triangle (see tri_adjacency, kernel/data_arrays.h) doesn't
   * return to the integrator's normal intersect/shade dispatch loop at
   * all -- it jumps straight to evaluating that neighbor's own prism, in
   * this same step, since exactly where the ray goes next is already
   * known and doesn't need a fresh BVH search to (re)discover. This is
   * what makes crossing many small triangles near a mesh's own sharp
   * corners cheap: those hops used to each cost a full closest-hit search
   * (self-exclusion tag write, BVH traversal, round-trip back out through
   * INTERSECT_CLOSEST) for something entirely predictable from the mesh's
   * own fixed topology. The loop below only falls through to that normal,
   * search-based path once a crossing's exit is the real surface, the
   * shell's own outer boundary, an unshared (mesh boundary) edge, or the
   * safety cap below is hit -- i.e. exactly the cases where "what's next"
   * genuinely isn't known in advance. */
  if (sd.type & PRIMITIVE_PRISM) {
    const ccl_global KernelObject &kobject = kernel_data_fetch(objects, sd.object);
    const float thickness = kobject.prism_shell_thickness;
    const float strength = kobject.prism_shell_strength;
    const float falloff_power = kobject.prism_falloff_power;
    const bool approximate = kobject.prism_approximate != 0;
    const float3 aura_color = make_float3(kobject.color[0], kobject.color[1], kobject.color[2]);

    Ray ray ccl_optional_struct_init;
    integrator_state_read_ray(state, &ray);

    /* Remember which object this crossing belongs to -- checked once this
     * path finally reaches a real surface, to discard the glow if it's this
     * same object's own surface (see prism_last_object in
     * state_template.h). */
    INTEGRATOR_STATE_WRITE(state, path, prism_last_object) = sd.object;

    int hop_prim = sd.prim;
    bool give_up_on_prisms = false;
    bool any_geometry = false;
    float last_t_enter = 0.0f;
    /* The very FIRST triangle's own entry point in this burst -- unlike
     * last_t_enter (which tracks wherever the chain currently is), this
     * one is trustworthy even if give-up fires: it's where the ray first
     * touched the shell, strictly before any hop-related drift could have
     * happened, so it's guaranteed not to have overshot the real surface
     * (see its use below). */
    float first_t_enter = 0.0f;
    /* Guards against a fast-hop chain making no real forward progress
     * along the ray. Confirmed via a diagnostic marker: two coplanar
     * triangles sharing a flat quad's diagonal can misclassify the exit
     * at that shared side plane (ray direction nearly parallel to it --
     * a near-zero-denominator case) and hop back and forth into each
     * other without ever actually advancing, silently burning through
     * all of PRISM_MAX_CROSSINGS. Worse, give-up's tmin advance (based on
     * whatever last_t_enter the oscillation left behind) could then
     * overshoot past the real surface, dropping the ray through to the
     * background entirely -- visible as a glowing seam exactly along
     * that diagonal. Scaled by thickness since t is a world-space ray
     * parameter and this effect's own spatial scale is the only
     * meaningful reference for "did this hop actually go anywhere".
     *
     * A genuinely coplanar shared edge (like a flat quad's diagonal) has
     * NO real forward distance to cross in the first place -- both
     * triangles compute essentially the same entry point for a ray near
     * that boundary, so any hop across it is inherently at the mercy of
     * pure floating-point noise, never a meaningful geometric crossing.
     * 1e-4 still let enough of that noise through to leave a faint
     * residual seam (still overshooting on a single accepted hop, without
     * needing to oscillate all the way to a full give-up); a real
     * lateral hop onto a genuinely different, differently-angled triangle
     * should clear a much larger bar than this even for a small mesh, so
     * this is bumped up two more orders of magnitude to reject that noise
     * more decisively. */
    const float min_hop_progress = thickness * 1e-2f;
    float prev_t_enter = -FLT_MAX;

    for (;;) {
      /* Safety backstop: normally a ray resolves (hits a real surface or
       * the background, or runs out of known neighbors to hop to) after a
       * small number of crossings, but this bounds the loop unconditionally
       * regardless of cause -- including a pathological/non-manifold mesh
       * where adjacency could otherwise chain far longer than any real
       * scene should need. Once this many crossings happen in a row
       * without resolving, give up on prisms entirely for the rest of this
       * burst (see PRISM_DISABLE_ALL_TAG below) rather than let it run
       * unbounded. */
      const uint16_t prism_crossings = INTEGRATOR_STATE(state, path, prism_crossings) + 1;
      INTEGRATOR_STATE_WRITE(state, path, prism_crossings) = prism_crossings;
      if (prism_crossings > PRISM_MAX_CROSSINGS) {
        give_up_on_prisms = true;
        break;
      }

      float3 tri_P[3];
      float3 tri_N[3];
      triangle_vertices_and_normals(kg, sd.object, hop_prim, tri_P, tri_N);

      float t_enter, t_exit;
      float3 Nface;
      int exit_plane;
      const bool has_geometry = (thickness > 0.0f) &&
                                ray_triangle_prism_intersect(ray.P,
                                                             ray.D,
                                                             ray.tmin,
                                                             ray.tmax,
                                                             tri_P[0],
                                                             tri_P[1],
                                                             tri_P[2],
                                                             tri_N[0],
                                                             tri_N[1],
                                                             tri_N[2],
                                                             thickness,
                                                             &t_enter,
                                                             &t_exit,
                                                             &Nface,
                                                             &exit_plane);
      if (!has_geometry) {
        /* Recomputed geometry test disagreed with whatever produced this
         * hit (should be rare, e.g. a hop landing exactly on a neighbor's
         * own boundary) -- resolve here rather than risk looping. */
        break;
      }

      if (t_enter < prev_t_enter + min_hop_progress) {
        /* No real forward progress since the last triangle in this chain
         * -- stop here and keep whatever the last GOOD entry point was
         * (any_geometry/last_t_enter are only updated below, once this
         * check passes), rather than accept this crossing and risk
         * hopping into an oscillation. Falls through to the normal
         * search-based path exactly like any other unresolved exit. */
        break;
      }
      prev_t_enter = t_enter;

      if (!any_geometry) {
        first_t_enter = t_enter;
      }
      any_geometry = true;
      last_t_enter = t_enter;

      if (strength > 0.0f) {
        /* Direct light: one NEE-style light sample at the entry point,
         * deliberately WITHOUT a shadow ray -- the aura is always treated
         * as lit, never occluded. Mirrors the light-selection/eval portion
         * of integrate_surface_direct_light() (this same file) up to (but
         * not including) shadow ray construction; no BSDF, so no MIS
         * weight either. Uses this crossing's own entry point, NOT sd.P --
         * sd.P is only valid for the very first triangle in a fast-hop
         * chain, since later ones are evaluated without rebuilding
         * ShaderData for them (that's the whole point of skipping the
         * normal dispatch loop). */
        const float3 current_P = ray.P + t_enter * ray.D;

        float3 light_rgb = zero_float3();
        {
          RNGState rng_state;
          path_state_rng_load(state, &rng_state);
          const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
          const uint bounce = INTEGRATOR_STATE(state, path, bounce);
          const float3 rand_light = path_state_rng_3D(kg, &rng_state, PRNG_LIGHT);

          LightSample ls ccl_optional_struct_init;
          /* SR_BSDF_HAS_TRANSMISSION tells the light tree/selection this
           * "receiver" needs light from every direction, not just Nface's
           * hemisphere -- without it, a face whose own triangle normal
           * points away from a light gets that light culled entirely
           * (zero contribution), creating a hard cutoff right where two
           * adjacent prism faces meet but only one is "lit" by hemisphere
           * logic. The aura has no real front/back, and per design should
           * pick up light regardless of any real geometry orientation or
           * occlusion in between. light_link_receiver_nee only reads
           * sd->object (light-linking is object-level, not per-triangle),
           * which stays correct throughout a fast-hop chain since all
           * hops stay within this same object. */
          if (light_sample_from_position(kg,
                                         rand_light,
                                         sd.time,
                                         current_P,
                                         Nface,
                                         light_link_receiver_nee(kg, &sd),
                                         SR_BSDF_HAS_TRANSMISSION,
                                         bounce,
                                         path_flag,
                                         &ls) &&
              ls.pdf > 0.0f)
          {
            Spectrum light_eval;
            const bool is_constant = light_sample_shader_eval_nee_constant(
                kg, ls.shader, ls.prim, ls.type != LIGHT_TRIANGLE, light_eval);
            if (!is_constant && ls.type != LIGHT_TRIANGLE && ls.type != LIGHT_BACKGROUND) {
              light_sample_shader_eval_forward(
                  kg, state, ls.prim, current_P, ls.D, ls.t, sd.time, light_eval);
            }
            light_rgb = spectrum_to_rgb(light_eval) * (ls.eval_fac / ls.pdf);
          }
        }

        /* Ambient: the world background's overall color, one-shot -- no
         * importance sampling, no recursive bounce (deliberately "not like
         * a diffuse surface"). Falls back to nothing for a non-constant
         * (textured/procedural) world; a research-prototype
         * simplification. */
        float3 ambient_rgb = zero_float3();
        {
          Spectrum ambient_eval;
          if (surface_shader_constant_emission(
                  kg, kernel_data.background.surface_shader, &ambient_eval))
          {
            ambient_rgb = spectrum_to_rgb(ambient_eval);
          }
        }

        /* Multiply then rescale: incoming light acts as a colored filter
         * over the aura's own inherent color; the magnitude comes entirely
         * from the density integral below, not from the light. */
        const float3 tinted = aura_color * (light_rgb + ambient_rgb);
        const float tinted_avg = average(tinted);
        const float3 hue = (tinted_avg > 1e-6f) ? tinted / tinted_avg : aura_color;

        /* Fog/volume-style magnitude: integrate the density along the
         * actual segment the ray travels through this prism, rather than
         * point-sampling at the entry -- see prism_density_integral's
         * comment in math_prism.h. */
        const float integral = prism_density_integral(
            ray.P, ray.D, tri_P[0], Nface, thickness, falloff_power, t_enter, t_exit);

        const Spectrum glow_L = rgb_to_spectrum(hue) * (strength * integral);

        /* Cap accumulated brightness at what this aura would look like AT
         * THE SURFACE (height = 0, i.e. density = 1, no falloff) --
         * strength * hue, with no integral scaling. No matter how many
         * prisms a ray crosses -- from the same object or several
         * overlapping ones -- the total can never exceed that fixed
         * ceiling, which is what makes this safe against runaway
         * summation blowing brightness out to white: overlapping
         * crossings can't sum past a value that doesn't depend on how
         * many of them there are.
         *
         * The cap is recomputed fresh from the hue of the prism being
         * crossed RIGHT NOW, not the first one this burst -- so if the ray
         * moves from one colored aura into a differently colored one, any
         * channel where the accumulated color would now exceed the new
         * prism's own cap gets pulled down to it. In practice the color
         * follows whichever prism is currently being crossed while the
         * overall brightness stays pinned at the same ceiling, like a
         * hand-off rather than two lights adding together. */
        const Spectrum surface_cap = rgb_to_spectrum(hue) * strength;
        /* Approximate mode (kobject.prism_approximate): combine crossings
         * with max() instead of summing them and relying on the cap above
         * -- a cheaper approximation that sidesteps brightness stacking
         * entirely (no crossing can ever brighten what's already there)
         * rather than letting contributions sum and clamping the result.
         * Still clamps a single crossing's own glow_L to the cap first, so
         * both modes share the same ceiling. */
        const Spectrum prism_prev = INTEGRATOR_STATE(state, path, prism_accum_L);
        const Spectrum prism_next = approximate ? max(prism_prev, min(glow_L, surface_cap)) :
                                                  min(prism_prev + glow_L, surface_cap);
        INTEGRATOR_STATE_WRITE(state, path, prism_accum_L) = prism_next;
      }

      if (exit_plane >= PRISM_EXIT_EDGE0) {
        const int edge_slot = exit_plane - PRISM_EXIT_EDGE0;
        const packed_int3 adj = kernel_data_fetch(tri_adjacency, hop_prim);
        const int neighbor = (edge_slot == 0) ? adj.x : (edge_slot == 1) ? adj.y : adj.z;
        if (neighbor >= 0) {
          /* Fast-hop: continue the loop with the known neighbor, same ray,
           * no BVH search. */
          hop_prim = neighbor;
          continue;
        }
      }

      /* Exit is the real surface, the shell's own outer boundary, or an
       * unshared (mesh boundary) edge -- "what's next" isn't known in
       * advance, so this burst resolves here and falls through to the
       * normal search-based path below. */
      break;
    }

    if (give_up_on_prisms) {
      /* Structurally exclude every prism, not just this one, from the very
       * next intersection search -- see PRISM_DISABLE_ALL_TAG in
       * math_prism.h and the matching check in kernel_embree_prism_
       * intersect_func_impl (device/cpu/bvh.h). Self-clearing: whatever
       * that next search finds overwrites isect.prim regardless, so this
       * only affects that one search, exactly the one that needs to land
       * on the real surface or background directly. */
      INTEGRATOR_STATE_WRITE(state, isect, prim) = PRISM_DISABLE_ALL_TAG;
    }
    else {
      /* Tag the LAST triangle processed this step's self-exclusion record
       * as "came from its prism", distinct from "came from its real
       * surface" -- both share the same prim index (see PRISM_SELF_TAG in
       * math_prism.h). Earlier triangles in a fast-hop chain need no such
       * tag: tmin (below) already advances past all of them (t is
       * non-decreasing along the chain, by construction of the shared
       * edge planes), so none of them can be rediscovered by the next
       * search regardless. */
      INTEGRATOR_STATE_WRITE(state, isect, prim) = hop_prim | PRISM_SELF_TAG;
    }

    /* Deliberately NOT calling path_state_next(): a ray that goes on to
     * hit this same object's real surface must shade exactly as if the
     * prism never existed, and path_state_next(LABEL_TRANSPARENT) sets
     * PATH_RAY_TRANSPARENT on the path flags, which changes how Russian
     * Roulette timing is computed for whatever comes next -- a real
     * behavioral difference, not a no-op. Returning LABEL_TRANSPARENT
     * directly still gets the desired dispatch (any non-terminating label
     * sends the path to INTERSECT_CLOSEST), while leaving path.bounce,
     * path.flag, transparent_bounce and rng_offset completely untouched.
     *
     * Advance minimally, just past the LAST processed triangle's own
     * entry point -- exactly the original always-safe strategy, just
     * applied to wherever the fast-hop chain actually ended up rather
     * than always the first triangle. This can never overshoot real
     * geometry (same guarantee as before), and correctly resumes right
     * where the chain left off without needing to re-discover any of the
     * triangles already handled internally above.
     *
     * Give-up is the one exception: last_t_enter reflects wherever a long,
     * possibly-degenerate hop chain ended up, which the forward-progress
     * guard above narrows but doesn't fully rule out (e.g. a chain that
     * creeps forward by just over the progress threshold each hop can
     * still run all the way to PRISM_MAX_CROSSINGS without ever properly
     * resolving). Rather than trust that potentially-drifted value, fall
     * back to first_t_enter -- the burst's very own starting point, which
     * by construction cannot be later than the real surface -- guaranteeing
     * give-up can never overshoot, at the cost of re-walking this same
     * burst's triangles via the normal search path instead of skipping
     * straight past them.
     *
     * NOTE: using first_t_enter unconditionally (not just for give-up) was
     * also tried, on the theory that any hopped-to triangle's own t_enter
     * could itself be individually corrupted, not just "insufficient
     * progress" (which the guard above already screens for). It did
     * remove the diagonal leak, but broke legitimate multi-triangle
     * grazing hops along the halo's silhouette -- visible as hard facet
     * creases where there used to be a smooth blend, plus a large
     * slowdown from the resulting redundant re-walking. Reverted; the
     * corruption this was trying to guard against needs a more targeted
     * fix than "distrust every hop unconditionally". */
    const float safe_t_enter = give_up_on_prisms ? first_t_enter : last_t_enter;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(
        any_geometry ? safe_t_enter : sd.ray_length);
    return LABEL_TRANSPARENT;
  }

  /* Prism shell (research prototype): this ray is done crossing prisms and
   * has reached a real surface -- shade it completely normally below (full
   * real shadows/direct/indirect, as if the prism never existed), then
   * flush whatever aura value it picked up crossing prisms on the way here
   * and ADD it on top (not a true alpha-overwrite -- Cycles' film buffer is
   * additive by nature, so a small contribution naturally lets the real
   * surface "be seen" through it rather than requiring the surface's own
   * already-written value to be modified). */
  {
    const Spectrum prism_total = INTEGRATOR_STATE(state, path, prism_accum_L);
    if (!is_zero(prism_total)) {
      /* Experiment: if this real surface belongs to the SAME object whose
       * prism shell the ray just came from, discard the accumulated glow
       * instead of flushing it -- landing back on that object's own
       * surface right after passing through its own shell was producing an
       * artifact. A different object's surface (or the background, handled
       * separately in shade_background.h) still gets the glow normally. */
      if (INTEGRATOR_STATE(state, path, prism_last_object) != sd.object) {
        film_write_surface_emission(kg, state, prism_total, 1.0f, render_buffer);
      }
      INTEGRATOR_STATE_WRITE(state, path, prism_accum_L) = zero_spectrum();
    }
    /* Reset the crossing-burst counter now that this burst resolved to a
     * real surface -- the cap bounds "crossings in a row before resolving
     * to something real", not the whole path's lifetime across further
     * bounces, so a legitimate multi-bounce path isn't penalized for
     * crossing a few prisms after every bounce. */
    INTEGRATOR_STATE_WRITE(state, path, prism_crossings) = 0;
    INTEGRATOR_STATE_WRITE(state, path, prism_last_object) = -1;
  }

  int continue_path_label = 0;

  const PathRayVisibility path_visibility = INTEGRATOR_STATE(state, path, visibility);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  /* Skip most work for volume bounding surface. */
#ifdef __VOLUME__
  if (!(sd.shader_flag & SD_HAS_ONLY_VOLUME)) {
#endif
#ifdef __SUBSURFACE__
    /* Can skip shader evaluation for BSSRDF exit point without bump mapping. */
    if (!(path_flag & PATH_RAY_SUBSURFACE) || ((sd.shader_flag & SD_HAS_BSSRDF_BUMP)))
#endif
    {
      /* Evaluate shader. */
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_EVAL);
      surface_shader_eval<node_feature_mask>(
          kg, state, &sd, render_buffer, path_visibility, path_flag);
    }

    if (sd.runtime_flag & SR_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

#ifdef __SPECTRAL__
    if (sd.runtime_flag & SR_BSDF_HAS_DISPERSION) {
      update_path_throughput_for_dispersion(kg, state, sd.rand_wavelength);
    }
#endif

    /* After shader evaluation, in case of texture cache miss. */
    guiding_record_surface_segment(kg, state, &sd);

#ifdef __SUBSURFACE__
    if (path_flag & PATH_RAY_SUBSURFACE) {
      /* When coming from inside subsurface scattering, setup a diffuse
       * closure to perform lighting at the exit point. */
      subsurface_shader_data_setup(kg, &sd);
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_SUBSURFACE;
    }
    else
#endif
    {
      /* Filter closures. */
      surface_shader_prepare_closures(kg, state, &sd, path_visibility);

      /* Evaluate holdout. */
      if (!integrate_surface_holdout(kg, state, &sd, render_buffer)) {
        return LABEL_NONE;
      }

      /* Write emission. */
      if (sd.runtime_flag & SR_EMISSION) {
        integrate_surface_emission(kg, state, &sd, render_buffer);
      }

      /* Perform path termination. Most paths have already been terminated in
       * the intersect_closest kernel, this is just for emission and for dividing
       * throughput by the probability at the right moment.
       *
       * Also ensure we don't do it twice for SSS at both the entry and exit point. */
      if (integrate_surface_terminate(state, path_flag)) {
        return LABEL_NONE;
      }

      /* Write render passes. */
#ifdef __PASSES__
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_PASSES);
      film_write_data_passes(kg, state, &sd, render_buffer);
#endif

#ifdef __DENOISING_FEATURES__
      film_write_denoising_features_surface(kg, state, &sd, render_buffer);
#endif
    }

    /* Load random number state. */
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
    if (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) {
      surface_shader_prepare_guiding(kg, state, &sd, &rng_state);
      guiding_write_debug_passes(kg, state, &sd, render_buffer);
    }
#endif
    /* Direct light. */
    PROFILING_EVENT(PROFILING_SHADE_SURFACE_DIRECT_LIGHT);
    const ShaderEvalResult result = integrate_surface_direct_light<node_feature_mask>(
        kg, state, &sd, &rng_state);
    if (result == SHADER_EVAL_CACHE_MISS) {
      return LABEL_CACHE_MISS;
    }

#if defined(__AO__)
    /* Ambient occlusion pass. */
    if (kernel_data.kernel_features & KERNEL_FEATURE_AO) {
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_AO);
      integrate_surface_ao(kg, state, &sd, &rng_state);
    }
#endif

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_bsdf_bssrdf_bounce(kg, state, &sd, &rng_state);
#ifdef __VOLUME__
  }
  else {
    if (integrate_surface_terminate(state, path_flag)) {
      return LABEL_NONE;
    }

#  ifdef __DENOISING_FEATURES__
    film_write_denoising_features_surface_volume(kg, state, &sd, render_buffer);
#  endif

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_volume_only_bounce(state, &sd);
  }

  if (continue_path_label & LABEL_TRANSMIT) {
    /* Enter/Exit volume. */
    volume_stack_enter_exit<false>(kg, state, &sd);
  }
#endif

  return continue_path_label;
}

template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_shade_surface_next_kernel(IntegratorState state)
{
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SUBSURFACE) {
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE);
  }
  else {
    kernel_assert(INTEGRATOR_STATE(state, ray, tmax) != 0.0f);
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
}

template<uint64_t node_feature_mask = KERNEL_FEATURE_NODE_MASK_SURFACE &
                                      ~KERNEL_FEATURE_NODE_RAYTRACE,
         DeviceKernel current_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE>
ccl_device_forceinline void integrator_shade_surface(KernelGlobals kg,
                                                     IntegratorState state,
                                                     ccl_global float *ccl_restrict render_buffer)
{
  const int continue_path_label = integrate_surface<node_feature_mask>(kg, state, render_buffer);
  if (continue_path_label == LABEL_CACHE_MISS) {
    integrator_path_cache_miss_sorted(state, current_kernel);
    return;
  }

#ifdef __MNEE__
  /* Cleanup MNEE flag and shadow path if it was not reused for shadow trace. */
  if ((kernel_data.kernel_features & KERNEL_FEATURE_MNEE) &&
      (INTEGRATOR_STATE(state, path, mnee) & PATH_MNEE_SAMPLED))
  {
    INTEGRATOR_STATE_WRITE(state, path, mnee) &= ~PATH_MNEE_SAMPLED;

    const IntegratorShadowState shadow_state = integrator_state_get_mnee_shadow_state(state);
    if (INTEGRATOR_STATE(shadow_state, shadow_path, queued_kernel) ==
        DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING)
    {
      integrator_shadow_path_terminate(shadow_state,
                                       DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING);
    }
  }
#endif

  if (continue_path_label == LABEL_NONE) {
    integrator_path_terminate(kg, state, render_buffer, current_kernel);
    return;
  }

#ifdef __SHADOW_LINKING__
  /* No need to cast shadow linking rays at a transparent bounce: the lights will be accumulated
   * via the main path in this case. BSSRDF bounces continue with intersect_subsurface. */
  if ((continue_path_label & (LABEL_TRANSPARENT | LABEL_SUBSURFACE_SCATTER)) == 0) {
    if (shadow_linking_schedule_intersection_kernel<current_kernel>(kg, state)) {
      return;
    }
  }
#endif

  integrator_shade_surface_next_kernel<current_kernel>(state);
}

ccl_device_forceinline void integrator_shade_surface_raytrace(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer)
{
  integrator_shade_surface<KERNEL_FEATURE_NODE_MASK_SURFACE,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE>(
      kg, state, render_buffer);
}

CCL_NAMESPACE_END
