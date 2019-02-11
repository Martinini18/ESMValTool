"""Derivation of variable `alb`.

authors:
    - crez_ba

"""

from iris import Constraint

from ._derived_variable_base import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `alb`."""

    # Required variables
    _required_variables = {
        'vars': [{
            'short_name': 'rsds',
            'field': 'T2{frequency}s'
        }, {
            'short_name': 'rsus',
            'field': 'T2{frequency}s'
        }]
    }

    def calculate(self, cubes):
        """Compute surface albedo."""
        rsds_cube = cubes.extract_strict(
            Constraint(name='surface_downwelling_shortwave_flux_in_air'))
        rsus_cube = cubes.extract_strict(
            Constraint(name='surface_upwelling_shortwave_flux_in_air'))

        rsns_cube = rsus_cube/rsds_cube 

        return rsns_cube
