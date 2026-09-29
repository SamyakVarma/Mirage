/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Prism shell (research prototype, see PRIMITIVE_PRISM in kernel/types.h):
 * ShaderData setup for a ray that hit the synthetic aura shell above a
 * JoySkin object's real surface, rather than the real surface itself.
 *
 * Stage 0 only needs enough here to satisfy shader_setup_from_ray's generic
 * downstream reads (backfacing test, shader_flag fetch) -- the actual
 * "shading" is a hardcoded solid-red bypass in shade_surface.h, so P/Ng/N
 * below don't need to be geometrically precise yet. */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline void prism_shader_setup(KernelGlobals kg,
                                          ccl_private ShaderData *ccl_restrict sd,
                                          const ccl_private Intersection *ccl_restrict isect,
                                          const ccl_private Ray *ccl_restrict ray)
{
  sd->P = ray->P + ray->D * isect->t;
  sd->Ng = -ray->D;
  sd->N = sd->Ng;
  sd->shader = 0;
  sd->dPdu = zero_float3();
  sd->dPdv = zero_float3();
}

CCL_NAMESPACE_END
