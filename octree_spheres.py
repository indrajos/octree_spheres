import numpy as np
import laspy
import matplotlib.pyplot as plt


las_filename = "2743_1234.las"


class OctreeNode:
    def __init__(self, center, size, points, max_points, depth=0):
        self.center = center  # Center of the node
        self.size = size  # Size of the node
        self.points = points  # Points contained in the node
        self.depth = depth  # Depth level in the octree
        self.max_points = max_points  # Maximum points per node
        self.children = []  # Children nodes
        self.sphere_radius = size / 2  # Sphere radius is half of node size
        self.sphere_center = center # Calculate the center of the sphere based on the node center


def las_points_to_list(las_filename):
    # Read LAS file
    las_data = laspy.read(las_filename)

    # Extract points
    points = np.vstack((las_data.x, las_data.y, las_data.z)).T

    # Create a list for each point
    points_list = []
    for i in range(len(points)):
        point_dict = {
            'x': points[i][0],
            'y': points[i][1],
            'z': points[i][2]
        }
        points_list.append(point_dict)

    return points_list


def create_octree(points_list, max_depth, max_points):
    # center detection
    min_x = min(point['x'] for point in points_list)
    max_x = max(point['x'] for point in points_list)
    min_y = min(point['y'] for point in points_list)
    max_y = max(point['y'] for point in points_list)
    min_z = min(point['z'] for point in points_list)
    max_z = max(point['z'] for point in points_list)

    # Calculate the world_size based on the minimum and maximum values
    world_size = {
        'x': max_x - min_x,
        'y': max_y - min_y,
        'z': max_z - min_z
    }

    # Create the root node of the octree using the calculated world_size
    octree_root = OctreeNode(center={'x': (max_x + min_x) / 2, 'y': (max_y + min_y) / 2, 'z': (max_z + min_z) / 2},
                              size=max(world_size.values()), points=points_list, max_points=max_points, depth=0)

    # call subdivide function
    subdivide(octree_root, max_depth)

    return octree_root, world_size


def inside_sphere(point, sphere_center, sphere_radius):
    point_coordinates = np.array([point['x'], point['y'], point['z']])
    sphere_center_coordinates = np.array([sphere_center['x'], sphere_center['y'], sphere_center['z']])

    # calculates the distance between point_coordinates and sphere_center_coordinates,
    # which gives the distance of the point from the center of the sphere
    distance = np.linalg.norm(point_coordinates - sphere_center_coordinates)

    # returns True if the calculated distance is less than or equal to the sphere_radius
    #  Points that are inside the sphere are included in the node's filtered_points
    return distance <= sphere_radius


# recursively subdividing an octree node into smaller nodes, until max_depth is reached or max_points_per_node is met
# function is called recursively on the newly created child node, continuing the process for each level of subdivision
def subdivide(node, max_depth):
    # checking if max_depth hasn't reached yet and ensures that the node has more points than max_points
    if node.depth < max_depth and len(node.points) > node.max_points:
        # node's size is divided by 2 to calculate the size of the child nodes
        sub_size = node.size / 2
        sub_centers = calculate_sub_centers(node.center, sub_size)

        # create list that will be added with points from the current node
        sub_points = [[] for _ in range(8)]
        # loop iterates point to find which child node should belong to the current node's center
        for point in node.points:
            sub_index = calculate_sub_index(point, node.center)
            sub_points[sub_index].append(point)

        # this loop selects the points that are inside the sphere associated with the child node
        for sub_center, sub_point_list in zip(sub_centers, sub_points):
            filtered_sub_points = [point for point in sub_point_list if
                                   inside_sphere(point, sub_center, sub_size / 2)]
            # creates new sub_node
            sub_node = OctreeNode(sub_center, sub_size, filtered_sub_points, node.max_points, node.depth + 1)
            node.children.append(sub_node)
            subdivide(sub_node, max_depth)


# it calculates the center positions of the eight child nodes when subdividing an octree node
# when the octree is split into eight smaller octants, each child node represents one of these octants
def calculate_sub_centers(center, sub_size):
    # a list to store the center positions of the eight child nodes
    sub_centers = []
    # i iterates for the x-axis for current dimension
    for i in range(2):
        # j iterates for the y-axis
        for j in range(2):
            # k iterates for the z-axis
            for k in range(2):
                # the value 0.5 is subtracted to shift their values from the integers 0 and 1
                # to the floating-point range of -0.5 and 0.5
                sub_center = {
                    'x': center['x'] + (i - 0.5) * sub_size,
                    'y': center['y'] + (j - 0.5) * sub_size,
                    'z': center['z'] + (k - 0.5) * sub_size
                }
                # an offset moves the sub-center position within the parent node
                sub_centers.append(sub_center)
    return sub_centers


# determine which child octant point belongs to within a parent octree node
def calculate_sub_index(point, center):
    index = 0
    # If the point's coordinate is greater than the center's coordinate,
    # that means the point is located on the positive side of the center dimension
    if point['x'] > center['x']:
        index |= 1
    if point['y'] > center['y']:
        index |= 2
    if point['z'] > center['z']:
        index |= 4
    # function returns the calculated sub_index, which is an integer ranging from 0 to 7
    return index


# visualize the current octree node as sphere
def visualize_octree_spheres(node, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # the center position of the sphere
    x, y, z = node.center['x'], node.center['y'], node.center['z']
    # radius of the sphere is calculated as half of the node's size
    sphere_radius = node.sphere_radius

    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    u, v = np.meshgrid(u, v)
    sphere_x = x + sphere_radius * np.sin(v) * np.cos(u)
    sphere_y = y + sphere_radius * np.sin(v) * np.sin(u)
    sphere_z = z + sphere_radius * np.cos(v)

    # Plot the sphere
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='blue', alpha=0.2)

    #  iterates through each child node of the current node and recursively calls itself on each child
    # that allows go through entire octree structure and visualize each node as a sphere
    for child in node.children:
        visualize_octree_spheres(child, ax)

    # returns the ax after all nodes have been added to it
    return ax


testing_points = las_points_to_list(las_filename)

max_depth = 20
max_points_per_node = 5

octree_r, world_size = create_octree(testing_points, max_depth, max_points_per_node)

# Calculate the overall size based on the world_size dimensions
world_size_values = list(world_size.values())
overall_size = max(world_size_values)

# Create subplot that includes the entire octree
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Set plot limits to cover the entire space based on overall_size
ax.set_xlim([octree_r.center['x'] - overall_size / 2, octree_r.center['x'] + overall_size / 2])
ax.set_ylim([octree_r.center['y'] - overall_size / 2, octree_r.center['y'] + overall_size / 2])
ax.set_zlim([octree_r.center['z'] - overall_size / 2, octree_r.center['z'] + overall_size / 2])

# The function recursively calls itself on each child node of the current node to visualize their positions and sizes
visualize_octree_spheres(octree_r, ax)

plt.show()
