#include "openmc/boundary_condition.h"

#include <exception>

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/surface.h"
#include "openmc/random_lcg.h" // [TESE] Obrigatório para o prn()
#include "openmc/settings.h" // [TESE] Para saber se é CE ou MG

namespace openmc {

//==============================================================================
// BoundaryCondition implementation [TESE - NOVA SEÇÃO]
//==============================================================================

//#include <cmath>
//#include <algorithm>

void BoundaryCondition::handle_albedo(Particle& p, const Surface& surf) const
{
  // -----------------------------------------------------------------------
  // [TESE] Lógicas Multigrupo (Vetor ou Matriz)
  // -----------------------------------------------------------------------
  if (has_mg_albedo_) {
    double E_in = p.E();
    int n_groups = albedo_energy_grid_.size() - 1;
    int g_in = -1;

    // 1. Localiza o grupo de energia do nêutron
    for (int i = 0; i < n_groups; ++i) {
      if (E_in >= albedo_energy_grid_[i] && E_in <= albedo_energy_grid_[i+1]) {
        g_in = i;
        break;
      }
    }

    if (g_in == -1) {
      p.wgt() = 0.0; 
      p.cross_vacuum_bc(surf);
      return;
    }

    // =====================================================================
    // LÓGICA 2: VETOR (Sem troca de grupo ou energia)
    // =====================================================================
    if (albedo_matrix_.size() == n_groups) {
      double prob_reflexao = albedo_matrix_[g_in];
      
      if (prn(p.seeds()) < prob_reflexao) {
        // SOBREVIVEU! A superfície já refletiu a direção dele.
        // Como não há troca de grupo, apenas limpamos o cache para segurança.
        p.material_last() = C_NONE;
        return; 
      } else {
        // PERDEU A ROLETA (Vazamento)
        p.wgt() = 0.0;
        p.cross_vacuum_bc(surf);
        return;
      }
    }

    // =====================================================================
    // LÓGICA 3: MATRIZ (Com troca de grupo de energia)
    // =====================================================================
    double xi = prn(p.seeds()); 
    double cumulative_prob = 0.0;
    int g_out = -1;

    for (int j = 0; j < n_groups; ++j) {
      cumulative_prob += albedo_matrix_[g_in * n_groups + j];
      if (xi < cumulative_prob) {
        g_out = j;
        break;
      }
    }

    if (g_out == -1) {
      p.wgt() = 0.0; 
      p.cross_vacuum_bc(surf);
      return;
    }

    // Atualiza a energia baseada na matriz
    double E_out_min = albedo_energy_grid_[g_out];
    double E_out_max = albedo_energy_grid_[g_out + 1];
    
    if (settings::run_CE) {
      double E_new = E_out_min + prn(p.seeds()) * (E_out_max - E_out_min);
      // Blindagem das tabelas S(a,b) e ENDF
      if (E_new < 1e-4) E_new = 1e-4;
      if (E_new > 19.5e6) E_new = 19.5e6;
      p.E() = E_new;
    } else {
      p.g() = g_out; 
      p.g_last() = p.g();
      p.E() = 0.5 * (E_out_min + E_out_max); 
    }

    // BLINDAGEM MÁXIMA DO VETOR DIRECIONAL (A causa secreta dos crashes)
    double norm = std::sqrt(p.u().x * p.u().x + p.u().y * p.u().y + p.u().z * p.u().z);
    if (norm > 0.0) {
      p.u().x /= norm;
      p.u().y /= norm;
      p.u().z /= norm;
    }

    p.material_last() = C_NONE;
    return;
  }

  // -----------------------------------------------------------------------
  // LÓGICA 1: PADRÃO ORIGINAL (Multiplicador Escalar de Peso)
  // -----------------------------------------------------------------------
  if (!has_albedo())
    return;

  double initial_wgt = p.wgt();
  p.wgt() *= (1.0 - albedo_);
  p.cross_vacuum_bc(surf);
  p.wgt() = initial_wgt * albedo_;
}
/*
void BoundaryCondition::handle_albedo(Particle& p, const Surface& surf) const
{
  // -----------------------------------------------------------------------
  // [TESE] Lógica Multigrupo
  // -----------------------------------------------------------------------
  if (has_mg_albedo_) {
    double E_in = p.E();
    int n_groups = albedo_energy_grid_.size() - 1;
    int g_in = -1;

    // 1. Acha o grupo de entrada
    for (int i = 0; i < n_groups; ++i) {
      if (E_in >= albedo_energy_grid_[i] && E_in <= albedo_energy_grid_[i+1]) {
        g_in = i;
        break;
      }
    }

    if (g_in == -1) {
      p.cross_vacuum_bc(surf);
      p.wgt() = 0.0;
      return;
    }

    // 2. Roleta da matriz de transferência
    double xi = prn(p.seeds()); 
    double cumulative_prob = 0.0;
    int g_out = -1;

    for (int j = 0; j < n_groups; ++j) {
      cumulative_prob += albedo_matrix_[g_in * n_groups + j];
      if (xi < cumulative_prob) {
        g_out = j;
        break;
      }
    }

    if (g_out == -1) {
      p.cross_vacuum_bc(surf);
      p.wgt() = 0.0;
      return;
    }

    // 3. Atualiza a energia
    double E_out_min = albedo_energy_grid_[g_out];
    double E_out_max = albedo_energy_grid_[g_out + 1];
    
    if (settings::run_CE) {
      double E_new = E_out_min + prn(p.seeds()) * (E_out_max - E_out_min);
      
      // PREVINE O CRASH DO LOG(0) EM SEÇÕES DE CHOQUE: Mantém a energia do nêutron estritamente dentro da tabela de dados
      if (E_new < 1e-4) {
        E_new = 1e-4;
      }
      if (E_new >= 19.9999e6) {
        E_new = 19.9999e6;
      }
      
      p.E() = E_new;
    }else{
      p.g() = g_out; // Atualiza o índice do grupo pro motor MGXS do OpenMC
      p.g_last() = p.g();
      // Mantém a coerência da energia média para os tallies do OpenMC não se perderem
      p.E() = 0.5 * (E_out_min + E_out_max);
    }

    // Força o OpenMC a recalcular as seções de choque macroscópicas para a nova energia
    // (Exatamente como o OpenMC faz internamente após uma colisão física).
    p.material_last() = C_NONE;
    return;
  }

  // -----------------------------------------------------------------------
  // Lógica Padrão (Escalar)
  // -----------------------------------------------------------------------
  if (!has_albedo())
    return;

  double initial_wgt = p.wgt();
  p.wgt() *= (1.0 - albedo_);
  p.cross_vacuum_bc(surf);
  p.wgt() = initial_wgt * albedo_;
}
*/

//==============================================================================
// VacuumBC implementation
//==============================================================================

void VacuumBC::handle_particle(Particle& p, const Surface& surf) const
{
  // Random ray and Monte Carlo need different treatments at vacuum BCs
  if (settings::solver_type == SolverType::RANDOM_RAY) {
    // Reflect ray off of the surface
    ReflectiveBC().handle_particle(p, surf);

    // Set ray's angular flux spectrum to vacuum conditions (zero)
    RandomRay* r = static_cast<RandomRay*>(&p);
    std::fill(r->angular_flux_.begin(), r->angular_flux_.end(), 0.0);

  } else {
    p.cross_vacuum_bc(surf);
  }
}

//==============================================================================
// ReflectiveBC implementation
//==============================================================================

void ReflectiveBC::handle_particle(Particle& p, const Surface& surf) const
{
  Direction u = surf.reflect(p.r(), p.u(), &p);
  u /= u.norm();

  // Handle the effects of the surface albedo on the particle's weight.
  BoundaryCondition::handle_albedo(p, surf);

  p.cross_reflective_bc(surf, u);
}

//==============================================================================
// WhiteBC implementation
//==============================================================================

void WhiteBC::handle_particle(Particle& p, const Surface& surf) const
{
  Direction u = surf.diffuse_reflect(p.r(), p.u(), p.current_seed());
  u /= u.norm();

  // Handle the effects of the surface albedo on the particle's weight.
  BoundaryCondition::handle_albedo(p, surf);

  p.cross_reflective_bc(surf, u);
}

//==============================================================================
// TranslationalPeriodicBC implementation
//==============================================================================

TranslationalPeriodicBC::TranslationalPeriodicBC(int i_surf, int j_surf)
  : PeriodicBC(i_surf, j_surf)
{
  Surface& surf1 {*model::surfaces[i_surf_]};
  Surface& surf2 {*model::surfaces[j_surf_]};

  // Make sure the first surface has an appropriate type.
  if (const auto* ptr = dynamic_cast<const SurfaceXPlane*>(&surf1)) {
  } else if (const auto* ptr = dynamic_cast<const SurfaceYPlane*>(&surf1)) {
  } else if (const auto* ptr = dynamic_cast<const SurfaceZPlane*>(&surf1)) {
  } else if (const auto* ptr = dynamic_cast<const SurfacePlane*>(&surf1)) {
  } else {
    throw std::invalid_argument(fmt::format(
      "Surface {} is an invalid type for "
      "translational periodic BCs. Only planes are supported for these BCs.",
      surf1.id_));
  }

  // Make sure the second surface has an appropriate type.
  if (const auto* ptr = dynamic_cast<const SurfaceXPlane*>(&surf2)) {
  } else if (const auto* ptr = dynamic_cast<const SurfaceYPlane*>(&surf2)) {
  } else if (const auto* ptr = dynamic_cast<const SurfaceZPlane*>(&surf2)) {
  } else if (const auto* ptr = dynamic_cast<const SurfacePlane*>(&surf2)) {
  } else {
    throw std::invalid_argument(fmt::format(
      "Surface {} is an invalid type for "
      "translational periodic BCs. Only planes are supported for these BCs.",
      surf2.id_));
  }

  // Compute the distance from the first surface to the origin.  Check the
  // surface evaluate function to decide if the distance is positive, negative,
  // or zero.
  Position origin {0, 0, 0};
  Direction u = surf1.normal(origin);
  double d1;
  double e1 = surf1.evaluate(origin);
  if (e1 > FP_COINCIDENT) {
    d1 = -surf1.distance(origin, -u, false);
  } else if (e1 < -FP_COINCIDENT) {
    d1 = surf1.distance(origin, u, false);
  } else {
    d1 = 0.0;
  }

  // Compute the distance from the second surface to the origin.
  double d2;
  double e2 = surf2.evaluate(origin);
  if (e2 > FP_COINCIDENT) {
    d2 = -surf2.distance(origin, -u, false);
  } else if (e2 < -FP_COINCIDENT) {
    d2 = surf2.distance(origin, u, false);
  } else {
    d2 = 0.0;
  }

  // Set the translation vector; it's length is the difference in the two
  // distances.
  translation_ = u * (d2 - d1);
}

void TranslationalPeriodicBC::handle_particle(
  Particle& p, const Surface& surf) const
{
  auto new_r = p.r() + translation_;
  int new_surface = p.surface() > 0 ? j_surf_ + 1 : -(j_surf_ + 1);

  // Handle the effects of the surface albedo on the particle's weight.
  BoundaryCondition::handle_albedo(p, surf);

  // Pass the new location and surface to the particle.
  p.cross_periodic_bc(surf, new_r, p.u(), new_surface);
}

//==============================================================================
// RotationalPeriodicBC implementation
//==============================================================================

RotationalPeriodicBC::RotationalPeriodicBC(
  int i_surf, int j_surf, PeriodicAxis axis)
  : PeriodicBC(std::abs(i_surf) - 1, std::abs(j_surf) - 1)
{
  Surface& surf1 {*model::surfaces[i_surf_]};
  Surface& surf2 {*model::surfaces[j_surf_]};

  // below convention for right handed coordinate system
  switch (axis) {
  case x:
    zero_axis_idx_ = 0; // x component of plane must be zero
    axis_1_idx_ = 1;    // y component independent
    axis_2_idx_ = 2;    // z component dependent
    break;
  case y:
    zero_axis_idx_ = 1; // y component of plane must be zero
    axis_1_idx_ = 2;    // z component independent
    axis_2_idx_ = 0;    // x component dependent
    break;
  case z:
    zero_axis_idx_ = 2; // z component of plane must be zero
    axis_1_idx_ = 0;    // x component independent
    axis_2_idx_ = 1;    // y component dependent
    break;
  default:
    throw std::invalid_argument(
      fmt::format("You've specified an axis that is not x, y, or z."));
  }

  Direction ax = {0.0, 0.0, 0.0};
  ax[zero_axis_idx_] = 1.0;

  auto i_sign = std::copysign(1, i_surf);
  auto j_sign = -std::copysign(1, j_surf);

  // Compute the surface normal vectors and make sure they are perpendicular
  // to the correct axis
  Direction norm1 = i_sign * surf1.normal({0, 0, 0});
  Direction norm2 = j_sign * surf2.normal({0, 0, 0});
  // Make sure both surfaces intersect the origin
  if (std::abs(surf1.evaluate({0, 0, 0})) > FP_COINCIDENT) {
    throw std::invalid_argument(fmt::format(
      "Rotational periodic BCs are only "
      "supported for rotations about the origin, but surface {} does not "
      "intersect the origin.",
      surf1.id_));
  }
  if (std::abs(surf2.evaluate({0, 0, 0})) > FP_COINCIDENT) {
    throw std::invalid_argument(fmt::format(
      "Rotational periodic BCs are only "
      "supported for rotations about the origin, but surface {} does not "
      "intersect the origin.",
      surf2.id_));
  }

  // Compute the signed rotation angle about the periodic axis. Note that
  // (n1×n2)·a = |n1||n2|sin(θ) and n1·n2 = |n1||n2|cos(θ), where a is the axis
  // of rotation.
  auto c = norm1.cross(norm2);
  angle_ = std::atan2(c.dot(ax), norm1.dot(norm2));

  // If the normals point in the same general direction, the surface sense
  // should change when crossing the boundary
  flip_sense_ = (i_sign * j_sign > 0.0);

  // Warn the user if the angle does not evenly divide a circle
  double rem = std::abs(std::remainder((2 * PI / angle_), 1.0));
  if (rem > FP_REL_PRECISION && rem < 1 - FP_REL_PRECISION) {
    warning(fmt::format(
      "Rotational periodic BC specified with a rotation "
      "angle of {} degrees which does not evenly divide 360 degrees.",
      angle_ * 180 / PI));
  }
}

void RotationalPeriodicBC::handle_particle(
  Particle& p, const Surface& surf) const
{
  int new_surface = p.surface() > 0 ? -(j_surf_ + 1) : j_surf_ + 1;
  if (flip_sense_)
    new_surface = -new_surface;

  // Rotate the particle's position and direction.
  Position r = p.r();
  Direction u = p.u();
  double cos_theta = std::cos(angle_);
  double sin_theta = std::sin(angle_);

  Position new_r;
  new_r[zero_axis_idx_] = r[zero_axis_idx_];
  new_r[axis_1_idx_] = cos_theta * r[axis_1_idx_] - sin_theta * r[axis_2_idx_];
  new_r[axis_2_idx_] = sin_theta * r[axis_1_idx_] + cos_theta * r[axis_2_idx_];

  Direction new_u;
  new_u[zero_axis_idx_] = u[zero_axis_idx_];
  new_u[axis_1_idx_] = cos_theta * u[axis_1_idx_] - sin_theta * u[axis_2_idx_];
  new_u[axis_2_idx_] = sin_theta * u[axis_1_idx_] + cos_theta * u[axis_2_idx_];

  // Handle the effects of the surface albedo on the particle's weight.
  BoundaryCondition::handle_albedo(p, surf);

  // Pass the new location, direction, and surface to the particle.
  p.cross_periodic_bc(surf, new_r, new_u, new_surface);
}

} // namespace openmc
