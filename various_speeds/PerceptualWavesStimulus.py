import PerceptualWavesAuxillaryFunctions as PWAux
import numpy as np
import math, random
from scipy.spatial import distance_matrix as dm
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from numba import jit, prange


@jit(nopython=True)
def _generate_flashing_disk(size_x, size_y, disk_pixels, disk_intensity, mean_luminance, frequency, t, dt):
    size_x = int(size_x)
    size_y = int(size_y) 

    image = np.ones((size_y, size_x), dtype=np.float32) * -1

    

    intensity = disk_intensity *  (np.sin(2 * np.pi * frequency * t * dt /1000))
    disk = (np.full((size_y, size_x), intensity, dtype=np.float32) + mean_luminance) * disk_pixels

    image_flat = image.ravel()
    disk_flat = disk.ravel()
    mask_flat = (disk_pixels == 1).ravel()
    image_flat[mask_flat] = disk_flat[mask_flat]
    image = image_flat.reshape(size_y, size_x)


    return image

@jit(nopython=True, parallel=True)
def generate_flashing_disk_sequence(size_x, size_y, disk_centre_eccentricity, theta, centre, distance_screen, screen_dimensions, radius_eccentricity,
                                     disk_intensity, mean_luminance, nt, temporal_frequency, dt):
    frames = np.zeros((nt, size_y, size_x), dtype=np.float32)
    disk_centre = PWAux._transform_eccentricity_theta_to_pixel_coordinates(disk_centre_eccentricity, theta, centre, distance_screen, screen_dimensions, size_x, size_y)


    radius = PWAux.radius_in_cm_from_eccentricity(disk_centre_eccentricity, radius_eccentricity, distance_screen, screen_dimensions, size_x, size_y)

    disk_pixels = PWAux.disk_from_centre_radius(disk_centre, radius, distance_screen, screen_dimensions, size_x, size_y)
    for i in prange(nt):
        t = i 
        frames[i] = _generate_flashing_disk(size_x, size_y,
                                             disk_pixels, disk_intensity, mean_luminance,
                                             temporal_frequency, t, dt)
    return frames

def create_cortical_surface(V1_area, distance_between_masses, distance_coef, sigma_s, nResponses, nLayers, ntheta, nt, coef_opo, coef_att, go, ga, gc, gr):
    area = 10e4
    length_side = np.sqrt(area)
    points_per_side = np.arange(0, length_side, distance_between_masses) 
    tuples_of_coordinates = np.array(np.meshgrid(points_per_side, points_per_side)).T.reshape(-1, 2) 

    vertices = PWAux.generate_polygon(center=(0.5, 0.5), avg_radius=0.2, side_ratio=6) 

    scaled_vertices_big = PWAux.scale_polygon_to_area(vertices, 2500)
    midpoint = PWAux.find_polygon_midpoint(scaled_vertices_big)

    vertices = PWAux.generate_polygon(center=(0.5, 0.5), avg_radius=0.2, side_ratio=6)
    scaled_vertices = PWAux.scale_polygon_to_area(vertices, V1_area)

    new_midpoint = PWAux.find_polygon_midpoint(scaled_vertices)
    for i in range(len(scaled_vertices)):
        scaled_vertices[i] = (scaled_vertices[i][0] + midpoint[0] - new_midpoint[0] + 45, scaled_vertices[i][1]  + midpoint[1] -  new_midpoint[1])



    points_in_polygon = PWAux.points_in_Polygon(scaled_vertices, tuples_of_coordinates)
    
    vertices_cortex = scaled_vertices
    neural_masses = points_in_polygon
    n_mass = len(points_in_polygon)

    distance_matrix = dm(neural_masses, neural_masses, p=distance_coef) 
    results = np.zeros([n_mass, nResponses, nLayers, ntheta, nt], dtype=np.float32)
    wo = go * np.exp(-distance_matrix**5 / sigma_s**coef_opo).astype(np.float32) # array of spatial spread of mutual inhibition 
    wa = ga * np.exp(-distance_matrix**5 / sigma_s**coef_att).astype(np.float32) # array of spatial spread of attentional modulation
    wc = gc * np.exp(-distance_matrix**5 / sigma_s**5).astype(np.float32) # array of spatial spread of colinear facilitation
    for idx in range(n_mass):
        wc[idx][idx] = 0
    wr = gr * np.exp(-distance_matrix**5 / sigma_s**5).astype(np.float32) # array of spatial spread of recurrent excitation
    wr_local = np.zeros_like(wr).astype(np.float32) # array of spatial spread of local recurrent excitation
    for idx in range(n_mass):
        wr_local[idx, idx] = wr[idx, idx]
    
    #initial condition: inject random imbalance at the initial time point. Here, it is done in terms of the imbalance between eyes. can do other random initiation.
    # results[:, 3, 0, :, 0] = np.random.uniform(size=1)*.2   
    # results[:, 3, 3, :, 0] = np.random.uniform(size=1)*.2


    return midpoint, vertices_cortex, neural_masses, n_mass, distance_matrix, results, wo, wa, wc, wr, wr_local


def make_PRFs(neural_masses, midpoint, stimulus_size, n_mass):
    scaled_neural_masses = np.array([neural_mass - midpoint for neural_mass in neural_masses])
    visual_field_pos_neural_masses = np.array([PWAux.cortical_to_visual_coordinates(neural_mass[0], neural_mass[1])[0] for neural_mass in scaled_neural_masses])
    eccentricity = np.array([PWAux.cortical_to_visual_coordinates(neural_mass[0], neural_mass[1])[1] for neural_mass in scaled_neural_masses])
    visual_field_pos_midpoint = np.array([0, 0])

    distances_visual_field_pos_neural_masses = np.array([np.sqrt((neural_mass[0] - visual_field_pos_midpoint[0])**2 + (neural_mass[1] - visual_field_pos_midpoint[1])**2) for neural_mass in visual_field_pos_neural_masses])

    radii_pRFs = np.array([PWAux.calculate_prf_radius(distance) for distance in distances_visual_field_pos_neural_masses])

    pRFs = PWAux.create_pRF_masks(stimulus_size, visual_field_pos_neural_masses, radii_pRFs, n_mass)

    return pRFs, visual_field_pos_neural_masses, eccentricity, radii_pRFs


def _compute_input_attentional(nt, pRFs, stimulus, dt, save_compute = True):
    
    num_prfs = len(pRFs)
    input = np.zeros((num_prfs, nt), dtype=np.float32)

    if save_compute:
        for t in range(int(100/dt)):
            max_input = np.max(stimulus[t] * pRFs, axis = (1,2))
            input[:, t] = max_input
        for seconds in range(int(100/dt), nt, int(100/dt)):
            input[:, seconds:seconds+int(100/dt)] = input[:, 0:int(100/dt)]

        
    else:
        for t in range(nt):
            max_input = np.max((stimulus[t] * pRFs), axis = (1,2))
            input[:, t] = max_input


    return input

