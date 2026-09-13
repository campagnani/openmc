#ifndef OPENMC_BOUNDARY_CONDITION_H
#define OPENMC_BOUNDARY_CONDITION_H

#include "openmc/hdf5_interface.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

// Forward declare some types used in function arguments.
class Particle;
class RandomRay;
class Surface;

//==============================================================================
//! A class that tells particles what to do after they strike an outer boundary.
//==============================================================================

class BoundaryCondition {
public:
  virtual ~BoundaryCondition() = default;

  //! Perform tracking operations for a particle that strikes the boundary.
  //! \param p The particle that struck the boundary.  This class is not meant
  //!   to directly modify anything about the particle, but it will do so
  //!   indirectly by calling the particle's appropriate cross_*_bc function.
  //! \param surf The specific surface on the boundary the particle struck.
  virtual void handle_particle(Particle& p, const Surface& surf) const = 0;

  //! Apply this boundary's albedo to a particle that has struck the surface.
  //!
  //! A scalar albedo reduces the particle weight and scores the lost weight as
  //! leakage. A multi-group albedo (group-wise vector or transfer matrix)
  //! classifies the incident energy on a user energy grid: the group-wise form
  //! uses the same weight reduction, while a transfer matrix additionally
  //! samples an outgoing energy group. Implementations of handle_particle
  //! typically call this method in its body.
  //!
  //! \param p The particle that struck the boundary.
  //! \param surf The specific surface on the boundary the particle struck.
  void handle_albedo(Particle& p, const Surface& surf) const;

  //! Return a string classification of this BC.
  virtual std::string type() const = 0;

  //! Write albedo data of this BC to hdf5.
  void to_hdf5(hid_t surf_group) const;

  //! Set a scalar albedo for this BC.
  void set_albedo(double albedo) { albedo_ = albedo; }

  //! Set a multi-group albedo defined on \p energy_grid.
  //!
  //! \p matrix must contain either N group-wise albedos or an N by N transfer
  //! matrix stored in row-major order, where N is energy_grid.size() - 1. The
  //! incident group index is the row.
  void set_mg_albedo(
    const vector<double>& energy_grid, const vector<double>& matrix);

  //! Return if this BC has a scalar albedo.
  bool has_albedo() const { return albedo_ > 0.0; }

  //! Return if this BC has a multi-group albedo.
  bool has_mg_albedo() const { return has_mg_albedo_; }

private:
  double albedo_ {-1.0};

  bool has_mg_albedo_ {false};
  vector<double> albedo_energy_grid_;
  vector<double> albedo_matrix_;
};

//==============================================================================
//! A BC that kills particles, indicating they left the problem.
//==============================================================================

class VacuumBC : public BoundaryCondition {
public:
  void handle_particle(Particle& p, const Surface& surf) const override;

  std::string type() const override { return "vacuum"; }
};

//==============================================================================
//! A BC that returns particles via specular reflection.
//==============================================================================

class ReflectiveBC : public BoundaryCondition {
public:
  void handle_particle(Particle& p, const Surface& surf) const override;

  std::string type() const override { return "reflective"; }
};

//==============================================================================
//! A BC that returns particles via diffuse reflection.
//==============================================================================

class WhiteBC : public BoundaryCondition {
public:
  void handle_particle(Particle& p, const Surface& surf) const override;

  std::string type() const override { return "white"; }
};

//==============================================================================
//! A BC that moves particles to another part of the problem.
//==============================================================================

class PeriodicBC : public BoundaryCondition {
public:
  PeriodicBC(int i_surf, int j_surf) : i_surf_(i_surf), j_surf_(j_surf) {};

  std::string type() const override { return "periodic"; }

  int i_surf() const { return i_surf_; }

  int j_surf() const { return j_surf_; }

protected:
  int i_surf_;
  int j_surf_;
};

//==============================================================================
//! A BC that moves particles to another part of the problem without rotation.
//==============================================================================

class TranslationalPeriodicBC : public PeriodicBC {
public:
  TranslationalPeriodicBC(int i_surf, int j_surf);

  void handle_particle(Particle& p, const Surface& surf) const override;

protected:
  //! Vector along which incident particles will be moved
  Position translation_;
};

//==============================================================================
//! A BC that rotates particles about a global axis.
//
//! Only rotations about the x, y, and z axes are supported.
//==============================================================================

class RotationalPeriodicBC : public PeriodicBC {
public:
  enum PeriodicAxis { x, y, z };
  RotationalPeriodicBC(int i_surf, int j_surf, PeriodicAxis axis);
  double compute_periodic_rotation(
    double rise_1, double run_1, double rise_2, double run_2) const;
  void handle_particle(Particle& p, const Surface& surf) const override;

protected:
  //! Angle about the axis by which particle coordinates will be rotated
  double angle_;
  //! Do we need to flip surfaces senses when applying the transformation?
  bool flip_sense_;
  //! Ensure that choice of axes is right handed. axis_1_idx_ corresponds to the
  //! independent axis and axis_2_idx_ corresponds to the dependent axis in the
  //! 2D plane  perpendicular to the planes' axis of rotation
  int zero_axis_idx_;
  int axis_1_idx_;
  int axis_2_idx_;
};

} // namespace openmc
#endif // OPENMC_BOUNDARY_CONDITION_H
