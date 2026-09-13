#include "openmc/boundary_condition.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/particle_type.h"
#include "openmc/random_lcg.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/surface.h"

namespace openmc {

namespace {

//! Reduce particle weight by the albedo and score the lost weight as leakage.
void apply_weight_albedo(Particle& p, const Surface& surf, double albedo)
{
  double initial_wgt = p.wgt();
  p.wgt() *= (1.0 - albedo);
  p.cross_vacuum_bc(surf);
  p.wgt() = initial_wgt * albedo;
}

//! Sample an energy in [eV] uniformly in lethargy within a group.
//!
//! The sampled energy is clipped to the loaded nuclear-data range so that
//! subsequent cross-section lookups remain on the union energy grid. Groups
//! whose lower bound is 0 eV are clipped to energy_min (typically 1e-5 eV).
double sample_ce_energy_in_group(double E_lo, double E_hi, uint64_t* seed)
{
  int neutron = ParticleType::neutron().transport_index();
  double data_lo = data::energy_min[neutron];
  double data_hi = data::energy_max[neutron];
  if (!(data_lo > 0.0) || !std::isfinite(data_lo))
    data_lo = 1.0e-5;
  if (!(data_hi > data_lo) || !std::isfinite(data_hi))
    data_hi = 20.0e6;

  E_lo = std::max(E_lo, data_lo);
  E_hi = std::min(E_hi, data_hi);
  if (!(E_lo > 0.0) || !std::isfinite(E_lo))
    E_lo = data_lo;
  if (!std::isfinite(E_hi) || E_hi <= E_lo) {
    return std::min(
      data_hi, std::nextafter(E_lo, std::numeric_limits<double>::infinity()));
  }

  double E_new =
    std::exp(std::log(E_lo) + prn(seed) * (std::log(E_hi) - std::log(E_lo)));
  if (!std::isfinite(E_new) || E_new < data_lo)
    E_new = data_lo;
  if (E_new > data_hi)
    E_new = data_hi;
  return E_new;
}

//! Return the albedo group containing E, or -1 if E is outside the grid.
//! Groups are [E_i, E_{i+1}) except the last group, which is closed.
int find_albedo_group(const vector<double>& grid, double E)
{
  int n_groups = static_cast<int>(grid.size()) - 1;
  if (n_groups <= 0 || !std::isfinite(E))
    return -1;
  if (E < grid.front() || E > grid.back())
    return -1;
  if (E == grid.back())
    return n_groups - 1;
  return static_cast<int>(upper_bound_index(grid.begin(), grid.end(), E));
}

} // namespace

//==============================================================================
// BoundaryCondition implementation
//==============================================================================

void BoundaryCondition::set_mg_albedo(
  const vector<double>& energy_grid, const vector<double>& matrix)
{
  if (energy_grid.size() < 2) {
    fatal_error("Multi-group albedo energy grid must contain at least two "
                "values.");
  }
  for (std::size_t i = 1; i < energy_grid.size(); ++i) {
    if (!(energy_grid[i] > energy_grid[i - 1])) {
      fatal_error("Multi-group albedo energy grid must be strictly "
                  "increasing.");
    }
  }

  std::size_t n_groups = energy_grid.size() - 1;
  if (matrix.size() != n_groups && matrix.size() != n_groups * n_groups) {
    fatal_error(
      fmt::format("Multi-group albedo must have {} entries (group-wise) or {} "
                  "entries (transfer matrix); got {}.",
        n_groups, n_groups * n_groups, matrix.size()));
  }

  bool any_gt_one = false;
  for (double v : matrix) {
    if (v < 0.0) {
      fatal_error("Multi-group albedo values must be non-negative.");
    }
    if (v > 1.0)
      any_gt_one = true;
  }
  if (any_gt_one) {
    warning("A multi-group albedo value is greater than 1, which may cause "
            "unphysical behaviour.");
  }

  albedo_energy_grid_ = energy_grid;
  albedo_matrix_ = matrix;
  has_mg_albedo_ = true;
}

void BoundaryCondition::to_hdf5(hid_t surf_group) const
{
  if (has_mg_albedo_) {
    write_dataset(surf_group, "albedo", albedo_matrix_);
    write_dataset(surf_group, "albedo_energy_grid", albedo_energy_grid_);
  } else if (has_albedo()) {
    write_string(surf_group, "albedo", fmt::format("{}", albedo_), false);
  }
}

void BoundaryCondition::handle_albedo(Particle& p, const Surface& surf) const
{
  if (has_mg_albedo_) {
    // Multi-group albedo is defined on a neutron energy grid.
    if (!p.type().is_neutron())
      return;

    int n_groups = static_cast<int>(albedo_energy_grid_.size()) - 1;
    int g_in = find_albedo_group(albedo_energy_grid_, p.E());
    if (g_in < 0 || g_in >= n_groups) {
      p.cross_vacuum_bc(surf);
      return;
    }

    // Group-wise vector: same weight-reduction treatment as a scalar albedo,
    // with no change to the particle energy.
    if (albedo_matrix_.size() == static_cast<std::size_t>(n_groups)) {
      apply_weight_albedo(p, surf, albedo_matrix_[g_in]);
      return;
    }

    // Transfer matrix, stored row-major as P(g_out | g_in). The row sum is the
    // survival probability; the outgoing group is sampled from the row.
    double row_sum = 0.0;
    for (int j = 0; j < n_groups; ++j)
      row_sum += albedo_matrix_[g_in * n_groups + j];

    apply_weight_albedo(p, surf, row_sum);
    if (!p.alive() || row_sum <= 0.0)
      return;

    uint64_t* seed = p.current_seed();
    double xi = prn(seed) * row_sum;
    double cumulative = 0.0;
    int g_out = n_groups - 1;
    for (int j = 0; j < n_groups; ++j) {
      cumulative += albedo_matrix_[g_in * n_groups + j];
      if (xi < cumulative) {
        g_out = j;
        break;
      }
    }

    double E_lo = albedo_energy_grid_[g_out];
    double E_hi = albedo_energy_grid_[g_out + 1];
    if (settings::run_CE) {
      p.E() = sample_ce_energy_in_group(E_lo, E_hi, seed);
    } else {
      p.E() = 0.5 * (E_lo + E_hi);
      p.g() = data::mg.get_group_index(p.E());
      p.g_last() = p.g();
    }

    // Force macroscopic cross sections to be rebuilt at the new energy.
    p.material_last() = C_NONE;
    return;
  }

  if (!has_albedo())
    return;

  apply_weight_albedo(p, surf, albedo_);
}

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
  if (!p.alive())
    return;

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
  if (!p.alive())
    return;

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
  if (!p.alive())
    return;

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
  if (!p.alive())
    return;

  // Pass the new location, direction, and surface to the particle.
  p.cross_periodic_bc(surf, new_r, new_u, new_surface);
}

} // namespace openmc
