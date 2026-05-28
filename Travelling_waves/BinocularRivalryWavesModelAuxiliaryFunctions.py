
import numpy as np
import math, random
from scipy.spatial import distance_matrix as dm
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from numba import jit
 
 
###################################################    Auxiliary Functions    #####################################

#########   mostly from https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
def generate_polygon(center: Tuple[float, float], 
                        avg_radius: float,
                        irregularity: float, 
                        spikiness: float,
                        num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on an ellips around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float): 
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float): 
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float): ##variance of the radial distances of each points 
            variance of the distance of each vertex to the center of
            the circumference. 
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check 
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")
    
    if num_vertices == 4:

        # Ensure regularity: Start at 45 degrees and step 90 degrees for a square
        initial_angle = math.pi / 4  # 45 degrees
        angle_step = 2 * math.pi / num_vertices
        angles = [initial_angle + i * angle_step for i in range(num_vertices)]

        # Generate square vertices
        points = [
            (
                center[0] + avg_radius * math.cos(angle),
                center[1] + avg_radius * math.sin(angle)
            )
            for angle in angles
        ]

        return points
        
    else:

        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = _random_angle_steps(num_vertices, irregularity)

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = _clip(random.gauss(avg_radius, spikiness), 0, 0.5)#2 * avg_radius
            point = (center[0] + 1.2 * radius * math.cos(angle), #to make an ellipse, not a circle
                    center[1] + radius * math.sin(angle))
            points.append(point)
            angle += angle_steps[i]

        return points


def _random_angle_steps(steps: int, 
                        irregularity: float) -> List[float]:
    """
    Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper) 
        angles.append(angle)
        cumsum += angle 

    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum 
    return angles 


def _clip(value, 
            lower, 
            upper): 
    """
    Given an interval, values outside the interval are clipped to the interval edges.

    Args:
        value (float): The value to be clipped.
        lower (float): The lower bound of the interval.
        upper (float): The upper bound of the interval.

    Returns:
        float: The clipped value.
    """
    return min(upper, max(value, lower)) 


def _calculate_polygon_area(vertices):
    """
    Calculate the signed area of a polygon using the Shoelace Formula.

    Args:
        vertices (List[Tuple[float, float]]): List of vertices, in CCW order. ##sort the vertices in clock wise direction 

    Returns:
        float: Signed area of the polygon.
    """
    n = len(vertices)
    area = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)

    area /= 2.0
    return abs(area)


def scale_polygon_to_area(vertices, 
                            target_area):
    """
    Scale a polygon to a target area.  

    Args:
        vertices (List[Tuple[float, float]]): List of vertices, in CCW order. 
        target_area (float): Target area for the scaled polygon.        

    Returns:
        List[Tuple[float, float]]: Scaled vertices of the polygon.
    """
    current_area = _calculate_polygon_area(vertices)

    if current_area == 0.0:
        raise ValueError("Polygon has zero area.")

    scaling_factor = math.sqrt(target_area / current_area)  

    scaled_vertices = [(x * scaling_factor, y * scaling_factor) for x, y in vertices]

    return scaled_vertices


def find_polygon_midpoint(vertices):
    """
    Find the midpoint (centroid) of a polygon.  

    Args:
        vertices (List[Tuple[float, float]]): List of vertices, in CCW order.

    Returns:
        Tuple[float, float]: Midpoint coordinates (x, y).
    """
    num_vertices = len(vertices)

    if num_vertices == 0:
        raise ValueError("Empty list of vertices.")

    x_sum = sum(x for x, _ in vertices)
    y_sum = sum(y for _, y in vertices)

    midpoint = np.array([x_sum / num_vertices, y_sum / num_vertices])

    return midpoint


def points_in_Polygon(vertices, 
                        squared_matrix_coordinates):
    """
    Find points that are inside a polygon.

    This function takes a list of vertices that define a polygon and a list of 
    coordinates representing points in a squared matrix. It returns a list of 
    points that are inside the polygon.

    Args:
    vertices (List[Tuple[float, float]]): List of vertices defining the polygon.
    squared_matrix_coordinates (List[Tuple[float, float]]): List of coordinates representing points in a squared matrix.

    Returns:
        List[List[float]]: List of points (as [x, y] pairs) that are inside the polygon.
    """

    polygon = Polygon(vertices) 
    points = []
    for pnt in squared_matrix_coordinates:
        pnt = Point(pnt)
        if polygon.contains(pnt):
            points.append(pnt)
    
    points_in_polygon = np.array([[point.x, point.y] for point in points])

    return points_in_polygon 




# According to Normal Visual Field. Jia Ma, Ning Fan, and Ningli Wang (2019), total binocular visual field of 210° horizontally and 130° vertically
# According to Cortical Magnification within Human Primary Visual Cortex Correlates with Acuity Thresholds. Duncan & Boynton, (2003), the power function fit for cortical magnification is: M = 9.81 × δ**−0.83


def _calculate_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points (x1, y1) and (x2, y2).
    
    Parameters:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.

    Returns:
    - distance: The Euclidean distance between the two points.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Adapts the original magnification function to any V1 area
def _magnification(sigma, simulated_V1, max_cortical_distance):
    """
    Cortical magnification function
    """
    return (9.81 * (sigma**-.83)) * (simulated_V1/max_cortical_distance)




def _find_eccentricity_for_cortical_position(target_M, simulated_V1, max_cortical_distance, step_size=0.01, max_eccentricity=155):
    
    # Initialize variables
    abs_target_M = abs(target_M)
    cumulative_distance = 0
    eccentricities = np.arange(0.1, max_eccentricity + step_size, step_size)

    # Loop through eccentricities and sum the cortical distances
    for e in eccentricities:
        cortical_distance = _magnification(e, simulated_V1, max_cortical_distance) * step_size
        cumulative_distance += cortical_distance

        # If cumulative distance exceeds or equals target_M, return the eccentricity
        if cumulative_distance >= abs_target_M:
            #invert signs of coordinates, so changing the side in the visual in repect to the cortex
            if target_M < 0:
                return e
            else:
                return -e

    return None # In case the target M value is beyond the provided range



def cortical_to_visual_coordinates(cortical_x, cortical_y, simulated_V1, max_cortical_distance, step_size=0.01, max_eccentricity=155):
    """
    Converts a point in the cortex (x, y in mm) to a point in the visual field (in degrees),
    accounting for quadrants and visual field inversion.
    """
    # Calculate the radial distance in the cortex
    radial_cortical_distance = _calculate_distance(cortical_x, cortical_y, 0, 0)

    # Get the scalar eccentricity corresponding to the radial cortical distance
    eccentricity = _find_eccentricity_for_cortical_position(radial_cortical_distance, simulated_V1, max_cortical_distance, step_size, max_eccentricity)

    if eccentricity is None:
        raise ValueError("Eccentricity not found within the given cortical range.")

    # Calculate the angle (theta) in radians for the cortical coordinates
    theta = np.arctan2(cortical_y, cortical_x)  # Angle in radians

    # Convert radial eccentricity to visual field coordinates using polar coordinates
    visual_x = eccentricity * np.cos(theta)
    visual_y = eccentricity * np.sin(theta)

    # Invert the Y-axis to account for the visual field inversion in the cortex
    visual_y = -visual_y

    return [visual_x, visual_y]


def calculate_prf_radius(eccentricity, scaling_factor=0.1, min_pRF_size=1):
    """
    Calculate receptive field radius as a function of eccentricity.
    
    Parameters:
    eccentricity (float): The eccentricity value (degrees).
    scaling_factor (float): Scaling factor to adjust the size of the receptive field.

    Returns:
    float: Calculated receptive field radius.
    """
    return round(((scaling_factor * eccentricity) + min_pRF_size) / 2, 2)  



def _transform_coordinates_visual_field_to_pixels(coordinates_in_visual_field, x_size_visual_field=210, y_size_visual_field=130, x_size_visual_stimulus=1920, y_size_visual_stimulus=1080):
    """
    Converts coordinates from visual field angular size (210x130 deg., according to Jia Ma, Ning Fan, and Ningli Wang (2019)) 
    into the input array (1920x1080 pix.) system, while centralizing both the visual field and stimulus ranges.

    Parameters:
    - coordinates_in_visual_field: x,y coordinates in visual field
    - x_size_visual_field: width of visual field in degrees of visual angle (default: 210)
    - y_size_visual_field: height of visual field in degrees of visual angle (default: 130)
    - x_size_visual_stimulus: width of the input array (default: 1920)
    - y_size_visual_stimulus: height of the input array (default: 1080)

    Returns:
    - (x_orig, y_orig): Transformed and centralized coordinates in the original system.
    """

    # Centralize visual field and stimulus coordinates
    x_visual_field_half = x_size_visual_field / 2  # Half width of visual field (105 deg)
    y_visual_field_half = y_size_visual_field / 2  # Half height of visual field (65 deg)

    # Transform the coordinates, and shift by half the stimulus size to ensure positive values
    x_orig = ((coordinates_in_visual_field[0] + x_visual_field_half) / x_size_visual_field) * x_size_visual_stimulus
    y_orig = ((coordinates_in_visual_field[1] + y_visual_field_half) / y_size_visual_field) * y_size_visual_stimulus

    return np.array([int(x_orig), int(y_orig)], dtype=np.float32)


def _transform_length_visual_field_to_pixels(length_in_degrees, x_size_visual_field=140, y_size_visual_field=140, x_size_visual_stimulus=1920, y_size_visual_stimulus=1080):# x_size_visual_field=210, y_size_visual_field=130,
    """
    Converts a length in degrees of visual field to pixels using a uniform scaling factor

    Parameters:
    - length_in_degrees: Length in degrees of visual field to be converted.
    - x_size_visual_field: Width of visual field in degrees (default: 210).
    - y_size_visual_field: Height of visual field in degrees (default: 130).
    - x_size_visual_stimulus: Width of the input array (default: 1920).
    - y_size_visual_stimulus: Height of the input array (default: 1080).

    Returns:
    - length_in_pixels: Converted length in pixels.
    """

    # Compute the scaling factor as the average scaling factor between the x and y dimensions
    scale_factor_x = x_size_visual_stimulus / x_size_visual_field
    scale_factor_y = y_size_visual_stimulus / y_size_visual_field
    average_scale_factor = (scale_factor_x + scale_factor_y) / 2

    # Convert length from degrees to pixels using the average scale factor
    length_in_pixels = length_in_degrees * average_scale_factor

    return int(length_in_pixels)


def create_pRF_masks(stimulus_size, visual_field_pos_neural_masses, radii_pRFs, n_mass):
    """
    Creates masks representing population receptive fields (pRFs) for each neural mass.
    Each mask will have values set to 1 within a circular area defined by a radius 
    around the center of the neural mass, and 0 elsewhere.

    Parameters:
    - stimulus_size: Tuple (width, height) representing the size of the stimulus (e.g., 1920x1080).
    - visual_field_pos_neural_masses: List of tuples containing the (x, y) positions of each neural mass 
      in the stimulus array.
    - radii_pRFs: List of radii for each neural mass that defines the size of the pRF in degrees of visual field.

    Returns:
    - pRFs: A 3D numpy array of shape (width, height, n_mass), where each layer corresponds to a mask
      for a different neural mass. The values in the mask are 1 inside the radius and 0 elsewhere.
    """
    
    pRFs = np.zeros([n_mass, stimulus_size[1], stimulus_size[0]], dtype=np.float32)  # Initialize the masks with zeros
    
    # Create a meshgrid for the coordinates of the array
    y_indices, x_indices = np.meshgrid(np.arange(stimulus_size[1]), np.arange(stimulus_size[0]))

    # Loop over each neural mass
    for number, (pos_neural_mass, radius) in enumerate(zip(visual_field_pos_neural_masses, radii_pRFs)):
        
        transformed_pos_neural_mass = _transform_coordinates_visual_field_to_pixels(pos_neural_mass)
        
        # Calculate the distance from each point in the array to the center of the current neural mass
        distance = _calculate_distance(x_indices, y_indices, transformed_pos_neural_mass[0], transformed_pos_neural_mass[1])
        
        # Set the points inside the radius to 1
        pRFs[number, :, :] = np.where(distance.T <= _transform_length_visual_field_to_pixels(radius), 1, 0)  # Transpose the distance matrix

    
    return pRFs

