import numpy as np
import math, random
from scipy.spatial import distance_matrix as dm
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from numba import jit


@jit(nopython=True)
def _transform_eccentricity_theta_to_pixel_coordinates(eccentricity, theta, centre, distance_screen, screen_dimensions, x_size, y_size):

    pixel_per_cm_x = x_size / screen_dimensions[0]
    pixel_per_cm_y = y_size / screen_dimensions[1]

    distance_from_centre = np.tan(eccentricity) * distance_screen
    
    x_coord_cm = distance_from_centre * np.cos(theta)
    y_coord_cm = distance_from_centre * np.sin(theta)

    pixel_coordinate_x = centre[0] + x_coord_cm * pixel_per_cm_x
    pixel_coordinate_y = centre[1] - y_coord_cm * pixel_per_cm_y

    return (pixel_coordinate_x, pixel_coordinate_y)

@jit(nopython=True)
def radius_in_cm_from_eccentricity(disk_centre_eccentricity, eccentricity, distance_screen, screen_dimensions, x_size, y_size):

    inner_radius = 122 * (np.tan(disk_centre_eccentricity) - np.tan(disk_centre_eccentricity - eccentricity))
    outer_radius = 122 * (np.tan(disk_centre_eccentricity + eccentricity) - np.tan(disk_centre_eccentricity))

    radius_in_cm = (inner_radius + outer_radius) / 2
    return radius_in_cm
    
@jit(nopython=True)
def disk_from_centre_radius(disk_coordinates, radius, distance_screen, screen_dimensions, x_size, y_size):

    x = np.arange(x_size).reshape(1, x_size) * np.ones((y_size, 1), dtype=np.float64)
    y = np.arange(y_size).reshape(y_size, 1) * np.ones((1, x_size), dtype=np.float64)
    pixel_per_cm_x = x_size / screen_dimensions[0]
    pixel_per_cm_y = y_size / screen_dimensions[1]

    radius_x = radius * pixel_per_cm_x
    radius_y = radius * pixel_per_cm_y

    mask = ((x - disk_coordinates[0])**2 / radius_x**2 + (y - disk_coordinates[1])**2 / radius_y**2) <= 1

    return mask

    
def generate_polygon(center, avg_radius, side_ratio):
# Ensure regularity: Start at 45 degrees and step 90 degrees for a square

    initial_angle = np.arctan(1 / side_ratio)
    angles = []
    angles.append(initial_angle)
    angles.append(initial_angle + (np.pi - 2* initial_angle))
    angles.append(initial_angle + np.pi)
    angles.append(initial_angle + np.pi + (np.pi - 2*initial_angle))

    # Generate square vertices
    points = [
        (
            center[0] + (avg_radius * math.cos(angle)),
            center[1] + (avg_radius * math.sin(angle))
        )
        for angle in angles
    ]
    return points

def scale_polygon_to_area(vertices, 
                            target_area):

    current_area = _calculate_polygon_area(vertices)

    if current_area == 0.0:
        raise ValueError("Polygon has zero area.")

    scaling_factor = math.sqrt(target_area / current_area)  

    scaled_vertices = [(x * scaling_factor, y * scaling_factor) for x, y in vertices]

    return scaled_vertices


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

def find_eccentricity(x_coord, y_coord):
    distance = np.sqrt(x_coord**2 + y_coord**2)
    e = 0.75 * np.exp(abs(distance) / 23.07) - 0.75
    return e 

def cortical_to_visual_coordinates(x_coord, y_coord):

    theta = np.arctan2(y_coord, x_coord)
    eccentricity = find_eccentricity(x_coord, y_coord)

    visual_x = eccentricity * np.cos(theta)
    visual_y = eccentricity * np.sin(theta)

    visual_y = -visual_y
    return [visual_x, visual_y], eccentricity



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


def _transform_coordinates_visual_field_to_pixels(coordinates_in_visual_field,  distance_from_screen = 122, screen_dimensions = (139, 77.5), size_x=960, size_y=540):
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
    px_per_cm_x = size_x / screen_dimensions[0]
    px_per_cm_y = size_y / screen_dimensions[1]

    radius_x_cm = distance_from_screen * np.tan((coordinates_in_visual_field[0] * np.pi / 180)) 
    radius_y_cm = distance_from_screen * np.tan((coordinates_in_visual_field[1] * np.pi / 180)) 

    radius_x_px = radius_x_cm * px_per_cm_x + (size_x / 2)
    radius_y_px = radius_y_cm * px_per_cm_y + (size_y / 2)



    return (radius_x_px, radius_y_px)

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

def _transform_length_visual_field_to_pixels(length_in_degrees, distance_from_screen = 122, screen_dimensions = (139, 77.5), size_x=960, size_y=540):# x_size_visual_field=210, y_size_visual_field=130,
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

    angle_x = np.arctan(screen_dimensions[0] / (2*distance_from_screen))
    angle_x = angle_x * 180 / np.pi
    px_per_deg_x = size_x / (2 * angle_x)
    angle_y = np.arctan(screen_dimensions[1] / (2*distance_from_screen))
    angle_y = angle_y * 180 / np.pi
    px_per_deg_y = size_y / (2 * angle_y)

    average_px_per_deg = (px_per_deg_x + px_per_deg_y)/2
    return average_px_per_deg * length_in_degrees


def create_pRF_masks(stimulus_size, visual_field_pos_neural_masses, radii_pRFs, n_mass):

    pRFs = np.zeros([n_mass, stimulus_size[1], stimulus_size[0]], dtype=np.float32) 
    y_indices, x_indices = np.meshgrid(np.arange(stimulus_size[1]), np.arange(stimulus_size[0]))

    for number, (pos_neural_mass, radius) in enumerate(zip(visual_field_pos_neural_masses, radii_pRFs)):
            
            transformed_pos_neural_mass = _transform_coordinates_visual_field_to_pixels(pos_neural_mass)
            
            # Calculate the distance from each point in the array to the center of the current neural mass
            distance = _calculate_distance(x_indices, y_indices, transformed_pos_neural_mass[0], transformed_pos_neural_mass[1])
            # Set the points inside the radius to 1
            pRFs[number, :, :] = np.where(distance.T <= _transform_length_visual_field_to_pixels(radius), 1, 0)  # Transpose the distance matrix

    
    return pRFs
        