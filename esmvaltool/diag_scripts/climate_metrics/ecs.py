#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to calculate ECS following Andrews et al. (2012).

Description
-----------
Calculate the equilibrium climate sensitivity (ECS) using the regression method
proposed by Andrews et al. (2012).

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
plot_ecs_regression : bool, optional (default: False)
    Plot the linear regression graph.
read_external_file : str, optional
    Read ECS from external file.

"""

import logging
import os
from pprint import pformat

import cf_units
import iris
import numpy as np
import yaml
from scipy import stats

from esmvaltool.diag_scripts.shared import (
    ProvenanceLogger, extract_variables, get_diagnostic_filename,
    get_plot_filename, group_metadata, io, plot, run_diagnostic,
    select_metadata, variables_available)

logger = logging.getLogger(os.path.basename(__file__))

EXP_4XCO2 = {
    'CMIP5': 'abrupt4xCO2',
    'CMIP6': 'abrupt-4xCO2',
}


def _get_anomaly_data(tas_data, rtmt_data, dataset, year_idx=None):
    """Calculate anomaly data for both variables."""
    project = tas_data[0]['project']
    exp_4xco2 = EXP_4XCO2[project]
    paths = {
        'tas_4x': select_metadata(tas_data, dataset=dataset, exp=exp_4xco2),
        'tas_pi': select_metadata(tas_data, dataset=dataset, exp='piControl'),
        'rtmt_4x': select_metadata(rtmt_data, dataset=dataset, exp=exp_4xco2),
        'rtmt_pi': select_metadata(
            rtmt_data, dataset=dataset, exp='piControl'),
    }
    ancestor_files = []
    cubes = {}
    for (key, [path]) in paths.items():
        ancestor_files.append(path['filename'])
        cube = iris.load_cube(path['filename'])
        cube = cube.aggregated_by('year', iris.analysis.MEAN)

        # Linear regression of piControl to account for drift and imbalance
        if key.endswith('pi'):
            reg = stats.linregress(cube.coord('year').points, cube.data)
            cube.data = reg.slope * cube.coord('year').points + reg.intercept

        # Extract correct years
        if year_idx is not None:
            cube = cube[year_idx]
        cubes[key] = cube

    # Check shapes
    shape = None
    for cube in cubes.values():
        if shape is None:
            shape = cube.shape
        else:
            if cube.shape != shape:
                raise ValueError(
                    "Expected all cubes of dataset '{}' to have identical "
                    "shapes, got {} and {}".format(dataset, shape, cube.shape))

    # Subtract linear trend of piControl experiments from abrupt 4xCO2
    cubes['tas_4x'].data -= cubes['tas_pi'].data
    cubes['rtmt_4x'].data -= cubes['rtmt_pi'].data
    return (cubes['tas_4x'], cubes['rtmt_4x'], ancestor_files)


def _get_multi_model_mean(cubes, project):
    """Get multi-model mean for dictionary of cubes."""
    if not cubes:
        logger.warning("Cannot calculate multi-model since no data was given.")
        return None
    mmm_data = []
    for cube in cubes.values():
        mmm_data.append(cube.data)
    cube = cubes[list(cubes.keys())[0]]
    mmm_data = np.ma.array(mmm_data)
    mmm_data = np.ma.mean(mmm_data, axis=0)
    mmm_cube = cube.copy(data=mmm_data)
    mmm_cube.attributes = {
        'dataset': 'MultiModelMean',
        'project': project,
    }
    return mmm_cube


def calculate_ecs(cfg, description, year_idx=None):
    """Calculate ECS and plot regression for a certain amount of years."""
    # Read external file if desired
    if cfg.get('read_external_file'):
        (ecs, clim_sens, external_file) = read_external_file(cfg)
    else:
        check_input_data(cfg)
        ecs = {}
        clim_sens = {}
        external_file = None

    # Read data
    input_data = cfg['input_data'].values()
    (tas_cubes, rtmt_cubes, ancestor_files) = preprocess_data(
        input_data, description, year_idx)

    # Iterate over all datasets and save ECS and climate sensitivity
    for (dataset, tas_cube) in tas_cubes.items():
        logger.info("Processing %s for %s", description, dataset)
        rtmt_cube = rtmt_cubes[dataset]
        ancestors = ancestor_files[dataset]

        # Plot ECS regression if desired
        (reg, path, provenance_record) = _plot_ecs_regression(
            cfg, dataset, description, tas_cube, rtmt_cube)

        # Provenance
        if path is not None:
            provenance_record['ancestors'] = ancestors
            with ProvenanceLogger(cfg) as provenance_logger:
                provenance_logger.log(path, provenance_record)

        # Save data
        if cfg.get('read_external_file') and dataset in ecs:
            logger.info(
                "Overwriting external given ECS and climate "
                "sensitivity for %s", dataset)
        ecs[dataset] = -reg.intercept / (2 * reg.slope)
        clim_sens[dataset] = -reg.slope

    # Write data
    tas_data = select_metadata(input_data, short_name='tas')
    rtmt_data = select_metadata(input_data, short_name='rtmt')
    ancestor_files = [d['filename'] for d in tas_data + rtmt_data]
    if external_file is not None:
        ancestor_files.append(external_file)
    write_data(ecs, clim_sens, ancestor_files, description, cfg)


def check_input_data(cfg):
    """Check input data."""
    if not variables_available(cfg, ['tas', 'rtmt']):
        raise ValueError("This diagnostic needs 'tas' and 'rtmt' "
                         "variables if 'read_external_file' is not given")
    input_data = cfg['input_data'].values()
    project_group = group_metadata(input_data, 'project')
    projects = list(project_group.keys())
    if len(projects) > 1:
        raise ValueError("This diagnostic supports only unique 'project' "
                         "attributes, got {}".format(projects))
    project = projects[0]
    if project not in EXP_4XCO2:
        raise ValueError("Project '{}' not supported yet".format(project))
    exp_group = group_metadata(input_data, 'exp')
    exps = set(exp_group.keys())
    if exps != {'piControl', EXP_4XCO2[project]}:
        raise ValueError("This diagnostic needs 'piControl' and '{}' "
                         "experiments, got {}".format(EXP_4XCO2[project],
                                                      exps))


def get_provenance_record(caption):
    """Create a provenance record describing the diagnostic data and plot."""
    record = {
        'caption': caption,
        'statistics': ['mean', 'diff'],
        'domains': ['global'],
        'authors': ['schl_ma'],
        'references': ['andrews12grl'],
        'realms': ['atmos'],
        'themes': ['phys'],
    }
    return record


def preprocess_data(input_data, description, year_idx=None):
    """Read input datasets and calculate anomalies and multi-model mean."""
    tas_data = select_metadata(input_data, short_name='tas')
    rtmt_data = select_metadata(input_data, short_name='rtmt')
    tas_cubes = {}
    rtmt_cubes = {}
    ancestor_files = {}
    project = tas_data[0]['project']

    # Calculate anomalies for every dataset
    for dataset in group_metadata(tas_data, 'dataset'):
        logger.info("Preprocessing %s for %s", description, dataset)
        (tas_cube, rtmt_cube, ancestors) = _get_anomaly_data(
            tas_data, rtmt_data, dataset, year_idx)
        tas_cubes[dataset] = tas_cube
        rtmt_cubes[dataset] = rtmt_cube
        ancestor_files[dataset] = ancestors

    # Calculate multi-model mean
    logger.info("Calculating multi-model mean")
    key = '{} MultiModelMean'.format(project)
    tas_cubes[key] = _get_multi_model_mean(tas_cubes, project)
    rtmt_cubes[key] = _get_multi_model_mean(rtmt_cubes, project)
    ancestor_files[key] = [d['filename'] for d in tas_data + rtmt_data]

    return (tas_cubes, rtmt_cubes, ancestor_files)


def read_external_file(cfg):
    """Read external file to get ECS."""
    ecs = {}
    clim_sens = {}
    if not cfg.get('read_external_file'):
        return (ecs, clim_sens)
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, cfg['read_external_file'])
    if os.path.isfile(filepath):
        with open(filepath, 'r') as infile:
            external_data = yaml.safe_load(infile)
    else:
        logger.error("Desired external file %s does not exist", filepath)
        return (ecs, clim_sens)
    ecs = external_data.get('ecs', {})
    clim_sens = external_data.get('climate_sensitivity', {})
    logger.info("External file %s", filepath)
    logger.info("Found ECS (K):")
    logger.info("%s", pformat(ecs))
    logger.info("Found climate sensitivities (W m-2 K-1):")
    logger.info("%s", pformat(clim_sens))
    return (ecs, clim_sens, filepath)


def _plot_ecs_regression(cfg, dataset_name, description, tas_cube, rtmt_cube):
    """Plot linear regression used to calculate ECS."""
    if not (cfg['write_plots'] and cfg.get('plot_ecs_regression')):
        return (None, None)
    reg = stats.linregress(tas_cube.data, rtmt_cube.data)

    # Regression line
    x_reg = np.linspace(-1.0, 9.0, 2)
    y_reg = reg.slope * x_reg + reg.intercept

    # Plot data
    title = '{} for {}'.format(description, dataset_name)
    plot_path = get_plot_filename(title.replace(' ', '_'), cfg)
    text = r'r = {:.2f}, $\lambda$ = {:.2f}, F = {:.2f}, ECS = {:.2f}'.format(
        reg.rvalue, -reg.slope, reg.intercept,
        -reg.intercept / (2.0 * reg.slope))
    plot.scatterplot(
        [tas_cube.data, x_reg],
        [rtmt_cube.data, y_reg],
        plot_path,
        plot_kwargs=[{
            'linestyle': 'none',
            'markeredgecolor': 'b',
            'markerfacecolor': 'none',
            'marker': 's',
        }, {
            'color': 'k',
            'linestyle': '-',
        }],
        save_kwargs={
            'bbox_inches': 'tight',
            'orientation': 'landscape',
        },
        axes_functions={
            'set_title': title,
            'set_xlabel': 'tas / ' + tas_cube.units.origin,
            'set_ylabel': 'rtmt / ' + rtmt_cube.units.origin,
            'set_xlim': [0.0, 8.0],
            'set_ylim': [-2.0, 10.0],
            'text': {
                'args': [0.05, 0.9, text],
                'kwargs': {
                    'transform': 'transAxes'
                },
            },
        },
    )

    # Write netcdf file for every plot
    netcdf_path = _save_evs_cube(tas_cube, rtmt_cube, reg, title, cfg)

    # Provenance
    provenance_record = get_provenance_record(
        "Scatterplot between TOA radiance and global mean surface temperature "
        "anomaly for {} of the abrupt 4x CO2 experiment including linear "
        "regression to calculate ECS for {} (following Andrews et al., "
        "Geophys. Res. Lett., 39, 2012).".format(description, dataset_name))
    provenance_record.update({
        'plot_file': plot_path,
        'plot_types': ['scatter'],
    })

    return (netcdf_path, provenance_record)


def _save_evs_cube(tas_cube, rtmt_cube, reg, title, cfg):
    """Save ECS cube for a given dataset."""
    ecs = -reg.intercept / (2.0 * reg.slope)
    tas_coord = iris.coords.AuxCoord(
        tas_cube.data,
        **extract_variables(cfg, as_iris=True)['tas'])
    attrs = {
        'model': title,
        'regression_r_value': reg.rvalue,
        'regression_slope': reg.slope,
        'regression_interception': reg.intercept,
        'climate_sensitivity': -reg.slope,
        'ECS': ecs,
    }
    cube = iris.cube.Cube(
        rtmt_cube.data,
        attributes=attrs,
        aux_coords_and_dims=[(tas_coord, 0)],
        **extract_variables(cfg, as_iris=True)['rtmt'])
    netcdf_path = get_diagnostic_filename(
        'ecs_regression_{}'.format(title.replace(' ', '_')), cfg)
    io.save_iris_cube(cube, netcdf_path)
    return netcdf_path


def write_data(ecs_data, clim_sens_data, ancestor_files, description, cfg):
    """Write netcdf files."""
    data = [ecs_data, clim_sens_data]
    var_attrs = [
        {
            'short_name': 'ecs',
            'long_name': 'Equilibrium Climate Sensitivity (ECS)',
            'units': cf_units.Unit('K'),
        },
        {
            'short_name': 'lambda',
            'long_name': 'Climate Sensitivity',
            'units': cf_units.Unit('W m-2 K-1'),
        },
    ]
    for (idx, var_attr) in enumerate(var_attrs):
        filename = '{}_{}'.format(var_attr['short_name'],
                                  description.replace(' ', '_'))
        path = get_diagnostic_filename(filename, cfg)
        attributes = {'Description': description}
        io.save_scalar_data(data[idx], path, var_attr, attributes=attributes)
        caption = ("{long_name} for multiple climate models for "
                   "{description}.".format(
                       description=description, **var_attr))
        provenance_record = get_provenance_record(caption)
        provenance_record['ancestors'] = ancestor_files
        with ProvenanceLogger(cfg) as provenance_logger:
            provenance_logger.log(path, provenance_record)


def main(cfg):
    """Run the diagnostic."""
    year_indices = {
        'all 150 years': None,
        'first 20 years': slice(None, 20),
        'last 130 years': slice(20, None),
    }
    for (descr, year_idx) in year_indices.items():
        logger.info("Considering %s for all datasets", descr)
        calculate_ecs(cfg, descr, year_idx)


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
