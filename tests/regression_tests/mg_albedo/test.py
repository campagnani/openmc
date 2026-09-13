import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture
def model():
    model = openmc.model.Model()

    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 4.5)
    model.materials.append(fuel)

    grid = [1.0e-5, 0.625, 20.0e6]
    # Two-group transfer matrix with downscatter from the fast group.
    albedo = [[0.90, 0.00],
              [0.10, 0.70]]
    x_min = openmc.XPlane(-5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    x_max = openmc.XPlane(5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    y_min = openmc.YPlane(-5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    y_max = openmc.YPlane(5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    z_min = openmc.ZPlane(-5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    z_max = openmc.ZPlane(5.0, boundary_type='reflective', albedo=albedo,
                          albedo_energy_grid=grid)
    cell = openmc.Cell(fill=fuel, region=(
        +x_min & -x_max & +y_min & -y_max & +z_min & -z_max))
    model.geometry = openmc.Geometry([cell])

    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 1000
    return model


def test_mg_albedo(model):
    harness = PyAPITestHarness('statepoint.10.h5', model)
    harness.main()
