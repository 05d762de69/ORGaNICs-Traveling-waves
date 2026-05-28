
import BinocularRivalryWavesModelAuxiliaryFunctions as brwmAux
import numpy as np
import math, random
from scipy.spatial import distance_matrix as dm
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from numba import jit

     


                        ################                     ################
                        ################    Create surface   ################
                        ################                     ################

 

def cortical_surface_model(
    nt, 
    nResponses,
    ntheta, 
    nLayers,
    go, 
    ga,
    gc,
    gr,
    sigma_s,
    distance_coef, 
    coef_opo, 
    coef_att, 
    V1_area=2500, # mean V1 size in human brain is 2100mm^2 acording to The Representation of the Visual Field in Human Striate Cortex A Revision of the Classic Holmes Map Jonathan C. Horton, William F. Hoyt, 1991
    cortical_surface=None
    ):
    """
    Create a random polygon representing a V1 cortical surface and instantiate neural masses inside this surface.

    Args:
        nt (int): Number of time steps.
        nResponses (int): Number of response types.
        ntheta (int): Number of theta orientations. 
        nLayers (int): Number of layers.
        go (float): Excitatory coupling strength.
        ga (float): Attentional coupling strength.
        sigma_s (float): Spatial spread parameter.
        distance_coef (int): Distance coefficient for calculating distance matrix.
        coef_opo (float): Coefficient for mutual inhibition.
        coef_att (float): Coefficient for attentional modulation. 
        V1_area (float): Area in mm^2 of desired cortical surface.
        cortical_surface (List[Tuple[float, float]]): List of vertices representing the desired cortical surface.

    Returns:
        Tuple: Tuple containing midpoint coordinates, cortex vertices, neural masses, number of masses, distance matrix, results array, excitatory array, and attentional array.
    """    
    #create a big square matrix with evenly spaced points 2.5mm distance
    area_size = 10e4 # area spanning 10000mm of potential cortical surface
    lenght_side_of_square = np.sqrt(area_size) 
    #mean width of one ocular-dominance column in human visual cortex v1 = 1.726 mm according to Heeger non-published paper
    distance_between_neural_masses = 2.5 # in mm, distance of a voxel in Computational neuroimaging and population receptive fields. Brian A. Wandell1 and Jonathan Winawer 2015
    points_per_side = np.arange(0, lenght_side_of_square, distance_between_neural_masses) 
    tuples_of_coordinates = np.array(np.meshgrid(points_per_side, points_per_side)).T.reshape(-1, 2) 
    
    if cortical_surface is None:
        
        # Generate polygon representing V1 cortical surface
        vertices = brwmAux.generate_polygon(center=(0.5, 0.5), avg_radius=0.2, irregularity=1, spikiness=0.05, num_vertices=100) #avg_radius=0.5, num_vertices=16, spikiness=0.2
        
        # Scale polygon to desired area 
        scaled_vertices = brwmAux.scale_polygon_to_area(vertices, V1_area) # mean V1 size in human brain is 2500mm^2 acording to The Representation of the Visual Field in Human Striate Cortex A Revision of the Classic Holmes Map Jonathan C. Horton, William F. Hoyt, 1991

    if cortical_surface == 'square':
        
        # Generate square representing V1 cortical surface
        vertices = brwmAux.generate_polygon(center=(0.5, 0.5), avg_radius=0.2, irregularity=0, spikiness=0, num_vertices=4) 
        
        # Scale square to desired area 
        scaled_vertices = brwmAux.scale_polygon_to_area(vertices, V1_area) # mean V1 size in human brain is 2500mm^2 acording to The Representation of the Visual Field in Human Striate Cortex A Revision of the Classic Holmes Map Jonathan C. Horton, William F. Hoyt, 1991
        
    else:
        
        scaled_vertices = cortical_surface 
    
    # Find points that are inside the polygon representing V1 cortical surface
    points_in_polygon = brwmAux.points_in_Polygon(scaled_vertices, tuples_of_coordinates)
    
    # Find midpoint of the polygon
    midpoint = brwmAux.find_polygon_midpoint(scaled_vertices)
    
    # Set proper variable names
    vertices_cortex = scaled_vertices
    neural_masses = points_in_polygon
    n_mass = len(points_in_polygon) 
    
    # Compute distance matrix between neural masses
    distance_matrix = dm(neural_masses, neural_masses, p=distance_coef)
    
    # Initialize arrays for results, inhibitory coupling, and attentional modulation
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


def make_pRFs(midpoint, 
              neural_masses, 
              n_mass, 
              stimulus_size=np.array([1920,1080])):
    """
    Create a population of population receptive fields (pRFs) for the model based on the neural mass positions in 
    the simulated cortical surface. The function converts neural mass positions to visual field positions 
    and generates pRFs using a magnification model. It also calculates the distances and radii of the pRFs 
    in respect to a central point (midpoint) in simulated cortical surface.

    Args:
    - midpoint: A tuple representing the central position (x, y) of the foveal representation in the simulated cortex (in mm).
    - neural_masses: A list of tuples containing the (x, y) coordinates of neural masses in the cortical surface (in mm).
    - n_mass: Integer representing the number of neural masses.
    - stimulus_size: A numpy array specifying the size of the stimulus (default is [1920, 1080]).

    Returns:
    - pRFs: A numpy array of shape (width, height, n_mass) containing the population receptive field (pRF) 
      masks for each neural mass. Each mask is represented as a 2D array of values, where values of 1 indicate 
      the area within the receptive field of a given neural mass.
    """
    
    # Finding the maximum x-coordinate of the simulated cortical surface. max_x/2 will be a scaling factor to adapt the magnification model to any size of simulated cortex
    max_x = max([pos[0] for pos in neural_masses]) - min([pos[0] for pos in neural_masses]) + 5 
    max_y = max([pos[1] for pos in neural_masses]) - min([pos[1] for pos in neural_masses]) + 5
    max_radial_distance = max([brwmAux._calculate_distance(pos[0], pos[1], 0, 0) for pos in neural_masses]) + 5

    # max_cortical_distance is the necessary cortical distance from the center of the foveal representation in the cortex to represent one visual hemifield (105°) using the original paramters Duncan & Boynton, (2003)
    max_cortical_distance_x = 88.62146920254175
    max_cortical_distance_y= 78.65853834016951
    
    # Re-center neural masses positions in respect to the midpoint
    scaled_neural_masses = np.array([neural_mass - midpoint for neural_mass in neural_masses])
    max_radial_distance = max([brwmAux._calculate_distance(pos[0], pos[1], 0, 0) for pos in scaled_neural_masses]) + 0
    
    # Finding visual field eccentricities for each neural mass position in the simulated cortex
    visual_field_pos_neural_masses = np.array([brwmAux.cortical_to_visual_coordinates(neural_mass[0], neural_mass[1], max_radial_distance, max_cortical_distance_x) for neural_mass in scaled_neural_masses])
    visual_field_pos_midpoint = np.array([0, 0])
  
    # Finding the distance in the visual field for each neural mass in respect to the visual field position of the midpoint
    distances_visual_field_pos_neural_masses = np.array([np.sqrt((neural_mass[0] - visual_field_pos_midpoint[0])**2 + (neural_mass[1] - visual_field_pos_midpoint[1])**2) for neural_mass in visual_field_pos_neural_masses])
    
    # Finding the pRFs for each neural mass in the simulated cortex
    radii_pRFs = np.array([brwmAux.calculate_prf_radius(distance) for distance in distances_visual_field_pos_neural_masses])
    
    # Create masks representing pRFs for each neural mass
    pRFs = brwmAux.create_pRF_masks(stimulus_size, visual_field_pos_neural_masses, radii_pRFs, n_mass)
    

    return pRFs, visual_field_pos_neural_masses




                        ################                     ################
                        ################    Create stimuli   ################
                        ################                     ################


def _generate_gabor_annulus(size_x, size_y, outer_radius, inner_radius, contrast, orientation, wavelength, phase, mean_luminance,
                        contrast_change_position, contrast_increment_amplitude, sigma_annulus, sigma_sector, dashed_line_position):
    
    size_x = int(size_x)
    size_y = int(size_y) 

    # Create an empty numpy array
    image = np.zeros((size_x, size_y), dtype=np.float32)

    # Create a meshgrid to represent the image coordinates
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))

    # Calculate the distance from the center of the image
    distance = np.sqrt((x - size_x / 2) ** 2 + (y - size_y / 2) ** 2)

    # Create a mask to select the annulus region
    annulus_mask = np.logical_and(distance >= inner_radius, distance <= outer_radius)
    annulus_mask_for_sector = np.logical_and(distance >= inner_radius*0.8, distance <= outer_radius*1.2)

    # Calculate the Gabor grating
    theta = orientation * np.pi / 180.0
    x_theta = np.cos(theta) * (x - size_x / 2) + np.sin(theta) * (y - size_y / 2)
    gabor = np.sin(2 * np.pi * x_theta / wavelength + phase)

    # Apply the contrast and mean luminance to the grating
    gabor = contrast * gabor + mean_luminance

    # Calculate the Gaussian envelopes for the annulus and the sector
    envelope_annulus = gaussian_filter(annulus_mask.astype(np.float32), sigma=sigma_annulus)
    angle = np.arctan2(y - size_y/2, x - size_x/2)
    angle = angle % (2 * np.pi)
    if contrast_change_position != 0:
        sector_mask = (angle >= (contrast_change_position-15)*(np.pi/180)) & (angle <= (contrast_change_position+15)*(np.pi/180))
    else:
        sector_mask = ((angle >= 0*(np.pi/180)) & (angle <= 15*(np.pi/180))) + ((angle >= 345*(np.pi/180)) & (angle <= 360*(np.pi/180)))
    envelope_sector = gaussian_filter(sector_mask.astype(np.float32), sigma=sigma_sector)

    # Apply the envelopes to the grating
    gabor_annulus = gabor * envelope_annulus
    gabor_sector = gabor * (1 + contrast_increment_amplitude * envelope_sector) * envelope_annulus

    # Combine the annulus and the sector with contrast increment
    gabor_annulus[annulus_mask_for_sector] = gabor_sector[annulus_mask_for_sector]

    # Insert the dashed lines that delimitate the target sector
    '''mask_dashed_line1 = (angle >= (dashed_line_position-18)*(np.pi/180)) & (angle <= (dashed_line_position-15)*(np.pi/180))
    for i in range(45):
        mask_dashed_line1[i::53] = False
    gabor_annulus[mask_dashed_line1] = -1

    if dashed_line_position != 360:
        mask_dashed_line2 = (angle >= (dashed_line_position+15)*(np.pi/180)) & (angle <= (dashed_line_position+18)*(np.pi/180))
        for i in range(45):
            mask_dashed_line2[i::53] = False
    else:
        mask_dashed_line2 = (angle >= 15*(np.pi/180)) & (angle <= 18*(np.pi/180))
        for i in range(45):
            mask_dashed_line2[i::53] = False
    gabor_annulus[mask_dashed_line2] = -1'''

    image = gabor_annulus
    image[image<-1] = -1
    image[image>1] = 1


    return image


def _create_gabor_annulus_stim(
    inner_radius,#np.array([[525, 309, 160], [382, 184, 144], [239, 59, 127]])
    outer_radius,#np.array([[525, 309, 160], [382, 184, 144], [239, 59, 127]])
    spatial_frequency=10,
    contrast_values=np.array([0.55, 0.45]),
    size_x=1920,
    size_y=1080,
    orientations=np.array([45,135]),
    angle_target_position=90,
    angle_contrast_increment=270,
    sigma_for_annulus= 11.25,
    sigma_for_sector= 11.25,    
    contrast_to_achieve_during_increment=0.95,
    contrast_to_achieve_during_increment_high_contrast_gabor=0.95
):
    
    inner_radius_pix = brwmAux._transform_length_visual_field_to_pixels(inner_radius)
    outer_radius_pix = brwmAux._transform_length_visual_field_to_pixels(outer_radius)
    spatial_frequency_pix = brwmAux._transform_length_visual_field_to_pixels(spatial_frequency)
    
    stimulus = np.array([
                _generate_gabor_annulus(size_x,size_y, outer_radius_pix, inner_radius_pix, 
                                    contrast_values[0], orientations[0], spatial_frequency_pix, 0, 0, 0, 0, sigma_for_annulus, 0, angle_target_position),
                _generate_gabor_annulus(size_x, size_y, outer_radius_pix, inner_radius_pix, 
                                    contrast_values[1], orientations[1], spatial_frequency_pix, 0, 0, 0, 0, sigma_for_annulus, 0, angle_target_position),
                _generate_gabor_annulus(size_x, size_y, outer_radius_pix, inner_radius_pix, 
                                    contrast_values[1], orientations[1], spatial_frequency_pix, 0, 0, angle_contrast_increment, contrast_to_achieve_during_increment/contrast_values[1], sigma_for_annulus, sigma_for_sector, angle_target_position),
                _generate_gabor_annulus(size_x,size_y, outer_radius_pix, inner_radius_pix, 
                                    contrast_values[0], orientations[0], spatial_frequency_pix, 0, 0, angle_contrast_increment, contrast_to_achieve_during_increment_high_contrast_gabor/contrast_values[0], sigma_for_annulus, sigma_for_sector, angle_target_position)
            ])
    
    
    return stimulus


#from numba import prange
#@jit(nopython=True, parallel=True)
def _compute_input(nt, pRFs, stimulus, pulse_time, condition):
    
    num_prfs = len(pRFs)
    input = np.zeros((num_prfs, 2, 2, nt), dtype=np.float32)

    # Compute the values once for stimulus[0] and stimulus[1]
    max_stim0 = np.max(abs(stimulus[0] * pRFs), axis=(1, 2))  # Compute for all pRFs at once

    max_stim1 = np.max(abs(stimulus[1] * pRFs), axis=(1, 2))  # Compute for all pRFs at once

    # Fill in the results for all time steps
    input[:, 0, 0, :] = max_stim0[:, np.newaxis]  # Broadcast the result to all time steps
    input[:, 1, 1, :] = max_stim1[:, np.newaxis]  # Broadcast the result to all time steps

    if condition == 'normal':
        
        # Special case for the pulse time 
        max_stim2 = np.max(abs(stimulus[2] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[0], pulse_time[1]):
            input[:, 1, 1, time_step] = max_stim2  # Update only the relevant time steps with max_stim2
            
    elif condition == 'double_pulse':
        
        # Special case for the pulse time - first eye
        max_stim2 = np.max(abs(stimulus[2] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[0], pulse_time[1]):
            input[:, 1, 1, time_step] = max_stim2  # Update only the relevant time steps with max_stim2
            
        # Special case for the pulse time - second eye
        max_stim3 = np.max(abs(stimulus[3] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[0], pulse_time[1]):
            input[:, 0, 0, time_step] = max_stim3  # Update only the relevant time steps with max_stim2
            
    elif condition == 'ping_pong':
        
        # Special case for the pulse time - first eye
        max_stim2 = np.max(abs(stimulus[2] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[0], pulse_time[1]):
            input[:, 1, 1, time_step] = max_stim2  # Update only the relevant time steps with max_stim2
            
        # Special case for the pulse time - second eye
        max_stim3 = np.max(abs(np.flip(stimulus[3]) * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[2], pulse_time[3]):
            input[:, 0, 0, time_step] = max_stim3  # Update only the relevant time steps with max_stim2
            
    elif condition == 'double_pulse2':
        
        # Special case for the pulse time - first eye
        max_stim2 = np.max(abs(stimulus[2] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[0], pulse_time[1]):
            input[:, 1, 1, time_step] = max_stim2  # Update only the relevant time steps with max_stim2
            
        # Special case for the pulse time - second eye
        max_stim3 = np.max(abs(stimulus[3] * pRFs), axis=(1, 2))  # Compute once for pulse times
        for time_step in range(pulse_time[2], pulse_time[3]):
            input[:, 0, 0, time_step] = max_stim3  # Update only the relevant time steps with max_stim2

    return input

    
#@jit(nopython=False)
def prepare_gabor_input(
    inner_radius,#np.array([[525, 309, 160], [382, 184, 144], [239, 59, 127]])
    outer_radius,#np.array([[525, 309, 160], [382, 184, 144], [239, 59, 127]])
    nt,
    pRFs,
    pulse_time=np.array([800,1000]),
    spatial_frequency=0.9,
    contrast_values=np.array([0.55, 0.45]),
    size_x=1920,
    size_y=1080,
    orientations=np.array([45,135]),
    angle_target_position=90,
    angle_contrast_increment=270,
    sigma_for_annulus= 11.25,
    sigma_for_sector= 11.25,    
    contrast_to_achieve_during_increment=0.95,
    contrast_to_achieve_during_increment_high_contrast_gabor=0.95,
    condition='normal'
):
        
        stimulus = _create_gabor_annulus_stim(inner_radius, outer_radius, spatial_frequency, contrast_values, size_x, size_y, orientations, angle_target_position, angle_contrast_increment, 
            sigma_for_annulus, sigma_for_sector, contrast_to_achieve_during_increment, contrast_to_achieve_during_increment_high_contrast_gabor)
        
        input = _compute_input(nt, pRFs, stimulus, pulse_time, condition)
        
        return input