/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/************************************ Path State *****************************/

KERNEL_STRUCT_BEGIN(path)
/* Index of a pixel within the device render buffer where this path will write its result.
 * To get an actual offset within the buffer the value needs to be multiplied by the
 * `kernel_data.film.pass_stride`.
 *
 * The multiplication is delayed for later, so that state can use 32bit integer. */
KERNEL_STRUCT_MEMBER(path, uint32_t, render_pixel_index, KERNEL_FEATURE_PATH_TRACING)
/* Current sample number. */
KERNEL_STRUCT_MEMBER(path, uint32_t, sample, KERNEL_FEATURE_PATH_TRACING)
/* Current ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current transparent ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, transparent_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current diffuse ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, diffuse_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current glossy ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, glossy_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current transmission ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, transmission_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current volume ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, volume_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current volume bounds ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, volume_bounds_bounce, KERNEL_FEATURE_PATH_TRACING)
/* Current portal ray bounce depth. */
KERNEL_STRUCT_MEMBER(path, uint16_t, portal_bounce, KERNEL_FEATURE_NODE_PORTAL)
/* DeviceKernel bit indicating queued kernels. */
KERNEL_STRUCT_MEMBER(path, uint16_t, queued_kernel, KERNEL_FEATURE_PATH_TRACING)
/* Random number generator per-pixel info. */
KERNEL_STRUCT_MEMBER(path, uint32_t, rng_pixel, KERNEL_FEATURE_PATH_TRACING)
/* Random number dimension offset. */
KERNEL_STRUCT_MEMBER(path, uint16_t, rng_offset, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayVisibilityFlag, limited to bits from PATH_RAY_VISIBILITY_ALL */
KERNEL_STRUCT_MEMBER(path, uint8_t, visibility, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayFlag */
KERNEL_STRUCT_MEMBER(path, uint32_t, flag, KERNEL_FEATURE_PATH_TRACING)
/* enum PathRayMNEE */
KERNEL_STRUCT_MEMBER(path, uint8_t, mnee, KERNEL_FEATURE_PATH_TRACING)
/* Index of shadow state path used for storing MNEE state. */
KERNEL_STRUCT_MEMBER(path, int, mnee_shadow_state, KERNEL_FEATURE_MNEE)
/* Majorant volume optical depth. */
KERNEL_STRUCT_MEMBER(path, float, optical_depth, KERNEL_FEATURE_PATH_TRACING)
/* Multiple importance sampling
 * The PDF of BSDF sampling at the last scatter point, which is at ray distance
 * zero and distance. Note that transparency and volume attenuation increase
 * the ray tmin but keep P unmodified so that this works. */
KERNEL_STRUCT_MEMBER(path, float, mis_ray_pdf, KERNEL_FEATURE_PATH_TRACING)
/* Object at last scatter point for light linking. */
KERNEL_STRUCT_MEMBER(path, int, mis_ray_object, KERNEL_FEATURE_LIGHT_LINKING)
/* Normal at last scatter point for light tree. */
KERNEL_STRUCT_MEMBER(path, packed_float3, mis_origin_n, KERNEL_FEATURE_PATH_TRACING)
/* Filter glossy. */
KERNEL_STRUCT_MEMBER(path, float, min_ray_pdf, KERNEL_FEATURE_PATH_TRACING)
/* Continuation probability for path termination. */
KERNEL_STRUCT_MEMBER(path,
                     FloatCompressedOnGPU,
                     continuation_probability,
                     KERNEL_FEATURE_PATH_TRACING)
/* Throughput. */
KERNEL_STRUCT_MEMBER(path, PackedSpectrum, throughput, KERNEL_FEATURE_PATH_TRACING)
/* Prism shell (research prototype, see PRIMITIVE_PRISM): running sum of
 * every prism crossing's density-integrated glow contribution along this
 * path, fog/volume-style -- each crossing adds its own segment integral
 * (see prism_density_integral in kernel/util/math_prism.h) regardless of
 * which object it belongs to, so overlapping shells naturally accumulate
 * brightness the way real overlapping fog would. Flushed to the film once
 * the path finally resolves to a real surface or the background (see
 * kernel/integrator/shade_surface.h / shade_background.h), not per
 * crossing. */
KERNEL_STRUCT_MEMBER(path, PackedSpectrum, prism_accum_L, KERNEL_FEATURE_PATH_TRACING)
/* Prism shell: number of prism crossings so far in the current unbroken
 * burst (reset once the burst resolves to a real surface or the
 * background -- see shade_surface.h / shade_background.h). Used purely as
 * a safety backstop: past PRISM_MAX_CROSSINGS (math_prism.h), the ray
 * gives up on prisms entirely for the rest of this burst via
 * PRISM_DISABLE_ALL_TAG, guaranteeing it actually resolves instead of
 * potentially needing an unbounded number of further crossings to clear a
 * dense overlap region. */
KERNEL_STRUCT_MEMBER(path, uint16_t, prism_crossings, KERNEL_FEATURE_PATH_TRACING)
/* Prism shell: object index of the most recently crossed prism shell along
 * this path (-1 if none yet). Checked when the path finally reaches a real
 * surface (shade_surface.h) or the background (shade_background.h): if that
 * surface belongs to this SAME object (or, for the background, if this
 * SAME object is the one the current ray segment departed from -- see
 * prism_depart_object below), the accumulated glow is discarded instead of
 * flushed to the film -- research-prototype experiment to suppress an
 * artifact from the shell's own glow stacking against that same object's
 * own surface shading. A different object's surface, or escaping to the
 * background without having departed from this object, still flushes
 * normally. */
KERNEL_STRUCT_MEMBER(path, int, prism_last_object, KERNEL_FEATURE_PATH_TRACING)
/* Prism shell: object index of the real surface the CURRENT ray segment
 * departed from (-1 for a fresh camera ray, which hasn't departed from any
 * object yet). Set whenever a real surface is shaded (shade_surface.h),
 * right before that shading spawns whatever ray(s) continue the path.
 *
 * This exists specifically for the background-hit case: a diffuse bounce
 * ray leaving a surface point right next to a sharp mesh edge very often
 * grazes back through that SAME object's own prism shell (the adjacent
 * face's shell is right there) before escaping to the background with no
 * further real-surface hit at all. shade_background.h has no "surface just
 * hit" to compare prism_last_object against the way shade_surface.h does,
 * so it uses this field instead -- otherwise that self-graze glow would
 * flush unconditionally regardless of prism_backface_culling, showing up as
 * a faint glow seam tracing the mesh's edges even with culling enabled. A
 * fresh camera ray's own prism-to-background crossing (prism_depart_object
 * still -1) is deliberately exempt, since that is the intended silhouette
 * rim glow, not a self-graze artifact. */
KERNEL_STRUCT_MEMBER(path, int, prism_depart_object, KERNEL_FEATURE_PATH_TRACING)
/* Factor to multiple with throughput to get remove any guiding PDFS.
 * Such throughput without guiding PDFS is used for Russian roulette termination. */
KERNEL_STRUCT_MEMBER(path, float, unguided_throughput, KERNEL_FEATURE_PATH_GUIDING)
/* Ratio of throughput to distinguish diffuse / glossy / transmission render passes. */
KERNEL_STRUCT_MEMBER(path,
                     SpectrumCompressedOnGPU,
                     pass_diffuse_weight,
                     KERNEL_FEATURE_LIGHT_PASSES)
KERNEL_STRUCT_MEMBER(path,
                     SpectrumCompressedOnGPU,
                     pass_glossy_weight,
                     KERNEL_FEATURE_LIGHT_PASSES)
/* Denoising. */
KERNEL_STRUCT_MEMBER(path,
                     SpectrumCompressedOnGPU,
                     denoising_feature_throughput,
                     KERNEL_FEATURE_DENOISING)
/* Shader sorting. */
/* TODO: compress as uint16? or leave out entirely and recompute key in sorting code? */
KERNEL_STRUCT_MEMBER(path, uint32_t, shader_sort_key, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_END(path)

/************************************** Ray ***********************************/

KERNEL_STRUCT_BEGIN_PACKED(ray, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, packed_float3, P, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, packed_float3, D, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, tmin, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, tmax, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, float, dP, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, FloatCompressedOnGPU, dD, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(ray, FloatCompressedOnGPU, time, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER(ray, float, previous_dt, KERNEL_FEATURE_LIGHT_TREE)
KERNEL_STRUCT_END(ray)

/*************************** Intersection result ******************************/

/* Result from scene intersection. */
KERNEL_STRUCT_BEGIN_PACKED(isect, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, t, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, u, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, float, v, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, prim, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, object, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_MEMBER_PACKED(isect, int, type, KERNEL_FEATURE_PATH_TRACING)
KERNEL_STRUCT_END(isect)

/*************** Subsurface closure state for subsurface kernel ***************/

KERNEL_STRUCT_BEGIN_PACKED(subsurface, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, PackedSpectrum, radius, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, packed_float3, N, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface, SpectrumCompressedOnGPU, albedo, KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_MEMBER_PACKED(subsurface,
                            FloatCompressedOnGPU,
                            anisotropy,
                            KERNEL_FEATURE_SUBSURFACE)
KERNEL_STRUCT_END(subsurface)

/********************************** Volume Stack ******************************/

KERNEL_STRUCT_BEGIN(volume_stack)
KERNEL_STRUCT_ARRAY_MEMBER(volume_stack, int, object, KERNEL_FEATURE_VOLUME)
KERNEL_STRUCT_ARRAY_MEMBER(volume_stack, int, shader, KERNEL_FEATURE_VOLUME)
KERNEL_STRUCT_END_ARRAY(volume_stack,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE,
                        KERNEL_STRUCT_VOLUME_STACK_SIZE)

/************************************ Path Guiding *****************************/
KERNEL_STRUCT_BEGIN(guiding)
#if defined(__PATH_GUIDING__)
/* Current path segment of the random walk/path. */
KERNEL_STRUCT_MEMBER(guiding,
                     openpgl::cpp::PathSegment *,
                     path_segment,
                     KERNEL_FEATURE_PATH_GUIDING)
#else
/* Current path segment of the random walk/path. */
KERNEL_STRUCT_MEMBER(guiding, uint64_t, path_segment, KERNEL_FEATURE_PATH_GUIDING)
#endif
/* If surface guiding is enabled */
KERNEL_STRUCT_MEMBER(guiding, bool, use_surface_guiding, KERNEL_FEATURE_PATH_GUIDING)
/* Random number used for additional guiding decisions (e.g., cache query, selection to use guiding
 * or BSDF sampling) */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     sample_surface_guiding_rand,
                     KERNEL_FEATURE_PATH_GUIDING)
/* The probability to use surface guiding (i.e., diffuse sampling prob * guiding prob). */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     surface_guiding_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
/* Probability of sampling a BSSRDF closure instead of a BSDF closure. */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     bssrdf_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
/* If volume guiding is enabled */
KERNEL_STRUCT_MEMBER(guiding, bool, use_volume_guiding, KERNEL_FEATURE_PATH_GUIDING)
/* Random number used for additional guiding decisions (e.g., cache query, selection to use guiding
 * or BSDF sampling) */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     sample_volume_guiding_rand,
                     KERNEL_FEATURE_PATH_GUIDING)
/* The probability to use surface guiding (i.e., diffuse sampling prob * guiding prob). */
KERNEL_STRUCT_MEMBER(guiding,
                     FloatCompressedOnGPU,
                     volume_guiding_sampling_prob,
                     KERNEL_FEATURE_PATH_GUIDING)
KERNEL_STRUCT_END(guiding)

/******************************* Shadow linking *******************************/

KERNEL_STRUCT_BEGIN(shadow_link)
/* Number of dedicated light hits along the shadow ray. */
KERNEL_STRUCT_MEMBER(shadow_link, uint16_t, dedicated_light_weight, KERNEL_FEATURE_SHADOW_LINKING)
/* Copy of primitive and object from the last main path intersection. */
KERNEL_STRUCT_MEMBER(shadow_link, int, last_isect_prim, KERNEL_FEATURE_SHADOW_LINKING)
KERNEL_STRUCT_MEMBER(shadow_link, int, last_isect_object, KERNEL_FEATURE_SHADOW_LINKING)
KERNEL_STRUCT_END(shadow_link)
