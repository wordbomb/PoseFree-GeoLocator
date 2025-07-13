import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import pyvista as pv

def visualize_simulation_scene_results2(flight, reconstruction_result, scale=1, output_file='simulations-scene.png', 
                                       point_color_range=(0, 1), quiver_length=40, 
                                       linewidth=1.0, x_lim=(-60, 120), y_lim=(-120, 90), z_lim=(0, 120),
                                       result_line_width=1, result_line_color='r'):
    
    plotter = pv.Plotter(off_screen=True)
    
    colors = np.linspace(point_color_range[0], point_color_range[1], flight.points.shape[1])
    point_cloud = pv.PolyData(flight.points.T)
    point_cloud['colors'] = colors
    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=10, scalars='colors', cmap='viridis')

    reconstruction_line = pv.lines_from_points(reconstruction_result.T)
    plotter.add_mesh(reconstruction_line, color='red', line_width=result_line_width)
    
    flight_line = pv.lines_from_points(flight.points.T)
    plotter.add_mesh(flight_line, color='red', line_width=result_line_width)

    for idx, camera in enumerate(flight.cameras):
        plotter.add_mesh(pv.Sphere(radius=2.0, center=camera.position), color='red')

        direction = scale * camera.R_camera2world[:, 0]  
        arrow = pv.Arrow(start=camera.position, direction=direction, scale=quiver_length)
        plotter.add_mesh(arrow, color='#D84B16')

        cube = pv.Cube(center=camera.position, x_length=8, y_length=8, z_length=4)
        plotter.add_mesh(cube, color='gray', opacity=0.5)
        
        plotter.add_point_labels([camera.position], [f'Camera {idx}'], font_size=12, text_color='black')

    plotter.set_background("white")
    plotter.add_axes()

    plotter.camera.clipping_range = (0.1, 5000)

    plotter.view_isometric()
    plotter.camera_position = [(x_lim[1], y_lim[1], z_lim[1]), (0, 0, 0), (0, 0, 1)]

    plotter.show()


def plot_rmse_distributions(all_rmse_lists, error_type='detection',title="", output_file="Reconstruction-Error.pdf",bbox_to_anchor=(1, 1),legend_show=True):
    """
    Plot RMSE distribution histograms for different error levels.

    Parameters:
    all_rmse_lists: List of tuples containing RMSE lists for different error levels
    error_type: Type of error ('detection' or 'camera_positioning')
    output_file: Filename to save the plot
    """
    plt.figure(figsize=(11, 6.5))
    colors = ['gray', 'blue', 'green', 'orange', 'purple', 'red']

    for i, (error, rmse_list) in enumerate(all_rmse_lists):
        if error_type == 'detection':
            label = f'Deviation level = {error} pixel'
        elif error_type == 'camera_positioning':
            label = f'Error level = {error} m'
        elif error_type == 'size_ratio':
            label = f'Scale factor = {error}'
        else:
            raise ValueError("Invalid error_type. Must be 'detection' or 'camera_positioning'.")

        counts, bins, patches = plt.hist(rmse_list, bins=50, color=colors[i], alpha=0.5, label=label, density=True)
        counts_percentage = counts
        for patch, count in zip(patches, counts_percentage):
            patch.set_height(count)
        mean_rmse = np.mean(rmse_list)
        plt.axvline(mean_rmse, color=colors[i], linestyle='dashed', linewidth=1)

    plt.xlabel(title+" of recognition (m)", fontsize=36)
    plt.ylabel("Frequency (%)", fontsize=36)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    if legend_show==True:
        plt.legend(fontsize=30, loc='upper right', bbox_to_anchor=bbox_to_anchor,handlelength=1.2,frameon=False)
    # plt.grid(True, alpha=0.1)
    plt.grid(False)

    # Adjust percentage formatting based on error type
    if error_type == 'detection':
        formatter = mticker.FuncFormatter(lambda x, _: f'{x / 1:.1f}%')
    elif error_type == 'camera_positioning':
        formatter = mticker.FuncFormatter(lambda x, _: f'{x / 10:.1f}%')
    elif error_type == 'size_ratio':
        formatter = mticker.FuncFormatter(lambda x, _: f'{x / 10:.1f}%')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def plot_rmse_distributions_detection(all_rmse_lists, output_file="Reconstruction-Error-detection.pdf"):
    plt.figure(figsize=(11, 6.5))
    colors = ['gray', 'blue', 'green', 'orange', 'purple', 'red']
    for i, (error, rmse_list) in enumerate(all_rmse_lists):
        counts, bins, patches = plt.hist(rmse_list, bins=50, color=colors[i], alpha=0.5, label=f'Detection error = {error} pixel', density=True)
        counts_percentage = counts
        for patch, count in zip(patches, counts_percentage):
            patch.set_height(count)
        mean_rmse = np.mean(rmse_list)
        plt.axvline(mean_rmse, color=colors[i], linestyle='dashed', linewidth=1)

    plt.xlabel("Reconstruction Error of Trajectory (meters)", fontsize=20)
    plt.ylabel("Frequency (%)", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18.4, loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.grid(True, alpha=0.1)

    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1:.1f}%'))
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def plot_rmse_distributions_camerapositioning(all_rmse_lists, output_file="Reconstruction-Error-camera-positioning.pdf"):
    plt.figure(figsize=(11, 6.5))
    colors = ['gray', 'blue', 'green', 'orange', 'purple', 'red']
    for i, (error, rmse_list) in enumerate(all_rmse_lists):
        counts, bins, patches = plt.hist(rmse_list, bins=50, color=colors[i], alpha=0.5, label=f'Camera positioning error = {error} m', density=True)
        counts_percentage = counts
        for patch, count in zip(patches, counts_percentage):
            patch.set_height(count)
        mean_rmse = np.mean(rmse_list)
        plt.axvline(mean_rmse, color=colors[i], linestyle='dashed', linewidth=1)

    plt.xlabel("Reconstruction Error of Trajectory (meters)", fontsize=20)
    plt.ylabel("Frequency (%)", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18.4, loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.grid(True, alpha=0.1)

    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 10:.1f}%'))
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def draw_cuboid(center, size):
    """
    Create a 3D cuboid at the specified center with the given size.
    
    Parameters:
    center: The center of the cuboid (x, y, z)
    size: The size of the cuboid (width, height, depth)
    """
    o = np.atleast_2d(center).astype(np.float64) - np.array(size, dtype=np.float64) / 2
    # List of vertices for the 6 faces of the cuboid
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
    v = v * np.array(size, dtype=np.float64)
    v += o
    faces = [[v[j] for j in [0, 1, 2, 3]], [v[j] for j in [4, 5, 6, 7]], 
             [v[j] for j in [0, 3, 7, 4]], [v[j] for j in [1, 2, 6, 5]], 
             [v[j] for j in [0, 1, 5, 4]], [v[j] for j in [2, 3, 7, 6]]]
    return faces


def draw_cuboid(center, size, rotation_matrix):
    """
    Create a 3D cuboid at the specified center with the given size and orientation defined by a rotation matrix.
    
    Parameters:
    center: The center of the cuboid (x, y, z)
    size: The size of the cuboid (width, height, depth)
    rotation_matrix: The rotation matrix defining the orientation of the cuboid
    """
    # Create vertices relative to the origin
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    v = (v - 0.5) * size  # Center the cuboid at origin and scale
    v_rotated = np.dot(v, rotation_matrix.T)  # Rotate the vertices
    v_translated = v_rotated + center  # Translate vertices to the center

    # Define the faces of the cuboid
    faces = [[v_translated[j] for j in [0, 1, 2, 3]], [v_translated[j] for j in [4, 5, 6, 7]], 
             [v_translated[j] for j in [0, 3, 7, 4]], [v_translated[j] for j in [1, 2, 6, 5]], 
             [v_translated[j] for j in [0, 1, 5, 4]], [v_translated[j] for j in [2, 3, 7, 6]]]
    return faces*5

def rotate_vector(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (cos_theta * v +
            sin_theta * np.cross(axis, v) +
            (1 - cos_theta) * np.dot(axis, v) * axis)

def draw_custom_quiver(ax, x, y, z, u, v, w, length=1.0, linewidth=1.5, alpha=1.0, headwidth=0.2, color='r', headlength=0.2,  angle=0):

    start = np.array([x, y, z])
    direction = np.array([u, v, w])
    
    direction = direction / np.linalg.norm(direction) * length
    
    end_main = start + direction
    end_adjusted = start + (length - headlength*length) * direction / np.linalg.norm(direction)
    

    perp_vector = np.cross(direction, np.array([0, 0, 1]))
    if np.linalg.norm(perp_vector) == 0:
        perp_vector = np.cross(direction, np.array([0, 1, 0]))
    perp_vector = perp_vector / np.linalg.norm(perp_vector) * headwidth * length

    perp_vector = rotate_vector(perp_vector, direction, np.radians(angle))

    base_tip_1 = end_adjusted + perp_vector
    base_tip_2 = end_adjusted - perp_vector

    ax.plot([base_tip_1[0], end_main[0]], [base_tip_1[1], end_main[1]], [base_tip_1[2], end_main[2]], linewidth=linewidth, color=color, alpha=alpha,solid_capstyle='round')
    ax.plot([base_tip_2[0], end_main[0]], [base_tip_2[1], end_main[1]], [base_tip_2[2], end_main[2]], linewidth=linewidth, color=color, alpha=alpha,solid_capstyle='round')

    ax.plot([start[0], end_main[0]], [start[1], end_main[1]], [start[2], end_main[2]],  linewidth=linewidth, color=color, alpha=alpha,solid_capstyle='round')


def visualize_simulation_scene(flight, elev=None, azim=None, scale=1, output_file='simulations-scene.pdf', point_alpha=0.5, 
                               point_marker='o', point_color_range=(0, 1), camera_marker='^', 
                               camera_color='r', quiver_length=40, linewidth=1.0, x_lim=(-60, 120), y_lim=(-120, 90)):
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.linspace(point_color_range[0], point_color_range[1], flight.points.shape[1])
    ax.scatter(flight.points[0], flight.points[1], flight.points[2], c=colors, marker=point_marker, alpha=point_alpha)

    for idx, camera in enumerate(flight.cameras):
        ax.scatter(*camera.position, c=camera_color, marker=camera_marker,s=0.1, label=f'Camera {idx}')
        if idx==2:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][0], scale * camera.R_camera2world[1][0], scale * camera.R_camera2world[2][0], 
                   length=quiver_length, color='#D84B16', angle=-50, linewidth=linewidth, alpha=0.8, headwidth=0.1, headlength=0.1)
        else:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][0], scale * camera.R_camera2world[1][0], scale * camera.R_camera2world[2][0], 
                   length=quiver_length, color='#D84B16', angle=90, linewidth=linewidth, alpha=0.8, headwidth=0.1, headlength=0.1)
        
        draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][1], scale * camera.R_camera2world[1][1], scale * camera.R_camera2world[2][1], 
                   length=quiver_length, color='#00B050', angle=30, linewidth=linewidth, alpha=0.8, headwidth=0.1, headlength=0.1)
        draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][2], scale * camera.R_camera2world[1][2], scale * camera.R_camera2world[2][2], 
                   length=quiver_length, color='#00B0F0', angle=30, linewidth=linewidth, alpha=0.8, headwidth=0.1, headlength=0.1)


        camera_size = [10, 10, 5]  # Adjust the size as needed
        faces = draw_cuboid(camera.position, camera_size, camera.R_camera2world)
        cuboid = Poly3DCollection(faces, facecolors='gray', linewidths=0.3, edgecolors='#7F7F7F', alpha=0)
        ax.add_collection3d(cuboid)
        # 添加文本标注
        if idx==0:
            ax.text(camera.position[0], camera.position[1]-20, camera.position[2]+15, f'Camera {idx}', color='black', fontsize=14)
        elif idx==1:
            ax.text(camera.position[0]-20, camera.position[1]-75, camera.position[2]-5, f'Camera {idx}', color='black', fontsize=14)
        elif idx==2:
            ax.text(camera.position[0], camera.position[1]-10, camera.position[2]+10, f'Camera {idx}', color='black', fontsize=14)

    ax.set_xlabel('X',fontsize=20, labelpad=10)
    ax.set_ylabel('Y',fontsize=20, labelpad=10)
    ax.set_zlabel('Z',fontsize=20, labelpad=0)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tick_params(axis='both', which='major', labelsize=18)
    if elev and azim:
        ax.view_init(elev=elev, azim=azim)
    # 使坐标轴背景透明
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.zaxis._axinfo['juggled'] = (1,2,0)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()



def visualize_simulation_scene_results(flight,reconstruction_result, elev=None, azim=None, scale=1, output_file='simulations-scene.pdf', point_alpha=0.5, 
                               point_marker='o', point_color_range=(0, 1), camera_marker='^', 
                               camera_color='r', quiver_length=40, linewidth=1.0, x_lim=(-60, 120), y_lim=(-120, 90), z_lim=(0, 120),
                               result_line_width=1,result_line_color='r',bbox_to_anchor=(1, 1)
                               ):
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.linspace(point_color_range[0], point_color_range[1], flight.points.shape[1])
    ax.scatter(reconstruction_result[0],reconstruction_result[1],reconstruction_result[2],label='Recognition', c=colors, marker=point_marker, alpha=point_alpha)
    ax.plot(flight.points[0], flight.points[1], flight.points[2],label='Ground-truth', color='r', linestyle='-',linewidth=result_line_width,solid_capstyle='round')

    for idx, camera in enumerate(flight.cameras):
        ax.scatter(*camera.position, c=camera_color, marker=camera_marker,s=0.1)


        if idx==2:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][0], scale * camera.R_camera2world[1][0], scale * camera.R_camera2world[2][0], 
                   length=quiver_length, color='#DF6F44', angle=-30, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        else:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][0], scale * camera.R_camera2world[1][0], scale * camera.R_camera2world[2][0], 
                   length=quiver_length, color='#DF6F44', angle=90, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        if idx==1:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][1], scale * camera.R_camera2world[1][1], scale * camera.R_camera2world[2][1], 
                   length=quiver_length, color='#33BF73', angle=0, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        else:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][1], scale * camera.R_camera2world[1][1], scale * camera.R_camera2world[2][1], 
                   length=quiver_length, color='#33BF73', angle=30, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        
        if idx==0:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][2], scale * camera.R_camera2world[1][2], scale * camera.R_camera2world[2][2], 
                   length=quiver_length, color='#33BFF3', angle=0, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        else:
            draw_custom_quiver(ax, 
                   camera.position[0], camera.position[1], camera.position[2], 
                   scale * camera.R_camera2world[0][2], scale * camera.R_camera2world[1][2], scale * camera.R_camera2world[2][2], 
                   length=quiver_length, color='#33BFF3', angle=30, linewidth=linewidth, alpha=1, headwidth=0.1, headlength=0.1)
        camera_size = [8, 8, 4]  # Adjust the size as needed
        faces = draw_cuboid(camera.position, camera_size, camera.R_camera2world)
        cuboid = Poly3DCollection(faces, facecolors='gray', linewidths=0.15, edgecolors='#C8C8C8', alpha=0)
        ax.add_collection3d(cuboid)
        # 添加文本标注
        if idx==0:
            ax.text(camera.position[0], camera.position[1]-20, camera.position[2]+23, f'Camera {idx}', color='black', fontsize=14,zorder=0,fontname='Verdana')
        elif idx==1:
            ax.text(camera.position[0]-20, camera.position[1]-75, camera.position[2]-5, f'Camera {idx}', color='black', fontsize=14,zorder=0,fontname='Verdana')
        elif idx==2:
            ax.text(camera.position[0], camera.position[1]-10, camera.position[2]+18, f'Camera {idx}', color='black', fontsize=14,zorder=0,fontname='Verdana')

    ax.set_xlabel('X',fontsize=20, labelpad=10,fontname='Verdana')
    ax.set_ylabel('Y',fontsize=20, labelpad=10,fontname='Verdana')
    ax.set_zlabel('Z',fontsize=20, labelpad=2,fontname='Verdana')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tick_params(axis='both', which='major', labelsize=18)
    if elev and azim:
        ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.zaxis._axinfo['juggled'] = (1,2,0)
    cmap = plt.get_cmap('viridis')
    color_steps = np.linspace(0, 1, 5)  # 5 gradient points
    gradient_handles = [Line2D([0], [0], linestyle='none', marker=point_marker, markersize=7, 
                                markerfacecolor=cmap(step), alpha=point_alpha) for step in color_steps]

    # Create a custom legend for the recognized line
    recognized_line = Line2D([0], [0], color='r', lw=result_line_width, label='Ground-truth')

    # Add the custom legend to the plot
    ax.legend([tuple(gradient_handles), recognized_line], ['Recognition', 'Ground-truth'], 
              handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper right',bbox_to_anchor=bbox_to_anchor,frameon=False,prop={'family': 'Verdana', 'size': 18})

    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def visualize_top_down_view(flight, point_color='b', point_marker='o', camera_color='r', camera_marker='^', 
                            x_label='X', y_label='Y', title='Top-Down View of Points and Cameras', grid=True, axis_equal=True):
    """
    Visualize the top-down view (2D projection).
    
    Parameters:
    flight: An object containing points and camera data
    point_color: Color of the spatial points, default is 'b' (blue)
    point_marker: Marker style for spatial points, default is 'o'
    camera_color: Color for camera positions, default is 'r' (red)
    camera_marker: Marker style for camera positions, default is '^'
    x_label: Label for the X-axis, default is 'X'
    y_label: Label for the Y-axis, default is 'Y'
    title: Title of the plot, default is 'Top-Down View of Points and Cameras'
    grid: Whether to display a grid, default is True
    axis_equal: Whether to keep the x-axis and y-axis scale equal, default is True
    """
    plt.figure()
    # Plot the top-down view of spatial points (assuming flight.points contains 3D point x, y coordinates at positions 0, 1)
    plt.scatter(flight.points[0], flight.points[1], c=point_color, marker=point_marker, label='Spatial Points')
    # Plot the top-down view of camera positions
    i = 0
    for camera in flight.cameras:
        plt.scatter(camera.position[0], camera.position[1], c=camera_color, marker=camera_marker, label=f'Camera {i}')
        i += 1
    
    # Set axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Set title
    plt.title(title)
    
    # Set scale and grid
    if axis_equal:
        plt.axis('equal')
    if grid:
        plt.grid(True)
    
    # Display the plot
    plt.legend()
    plt.show()

def visualize_camera_projection(flight, camera_id=0, image_width=1920, image_height=1080, 
                                output_file='Projection-of-Spatial-Points-on-Camera-0-Image-Plane.pdf', 
                                color_start=0, color_end=1, marker_size=8):
    """
    Visualize the camera projection image.
    
    Parameters:
    flight: An object containing points and camera data
    camera_id: The ID of the camera to view, default is 0
    image_width: Width of the image, default is 1920
    image_height: Height of the image, default is 1080
    output_file: Filename to save the image, default is 'Projection-of-Spatial-Points-on-Camera-0-Image-Plane.pdf'
    color_start: Starting value for the color gradient, default is 0
    color_end: Ending value for the color gradient, default is 1
    marker_size: Size of the markers, default is 8
    """
    # Create a color gradient
    colors = np.linspace(color_start, color_end, flight.points.shape[1])
    
    # Create the image
    plt.figure(figsize=(11, 7))
    ax = plt.gca()  # Get current axis
    
    ax.scatter(flight.cameras[camera_id].imagepoints.T[1],
               flight.cameras[camera_id].imagepoints.T[2],
               c=colors, 
               marker='o', 
               s=marker_size)
    
    # Set axis limits
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    
    # Set axis labels and tick sizes
    ax.set_xlabel('u', fontsize=60)
    ax.set_ylabel('v', fontsize=60, rotation=0)  # 'rotation=0' makes 'v' vertical
    ax.tick_params(axis='both', which='major', labelsize=32)
    
    # Invert the y-axis
    ax.invert_yaxis()
    
    # Move the x-axis to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    # Move the y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right') 

    # Set custom axis ticks
    
    ax.set_xticks([i for i in range(0, image_width+1, int(image_width/6))])
    ax.set_yticks([i for i in range(0, image_height+1, int(image_height/4))])

    # Thicken the border lines
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)  # Adjust the value to make the border thicker or thinner
    
    # Save the image
    plt.savefig(output_file, bbox_inches='tight')
    
    # Display the image
    plt.show()

def visualize_3d_scatter(flight, result, title='Circle Points', point_marker='o', point_color_range=(0, 1), 
                         result_marker='o', result_color='b', x_label='X', y_label='Y', z_label='Z'):
    """
    Visualize a 3D scatter plot.
    
    Parameters:
    flight: An object containing points and camera data
    result: The coordinates of the computed points
    title: The title of the plot, default is 'Circle Points'
    point_marker: Marker style for spatial points, default is 'o'
    point_color_range: Range of colors for spatial points, default is (0, 1)
    result_marker: Marker style for the computed points, default is 'o'
    result_color: Color for the computed points, default is 'b' (blue)
    x_label: Label for the X-axis, default is 'X'
    y_label: Label for the Y-axis, default is 'Y'
    z_label: Label for the Z-axis, default is 'Z'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colors for points
    colors = np.linspace(point_color_range[0], point_color_range[1], flight.points.shape[1])
    
    # Plot spatial points
    ax.scatter(flight.points[0], flight.points[1], flight.points[2], c=colors, marker=point_marker)
    
    # Plot computed points
    ax.scatter(result[0], result[1], result[2], c=result_color, marker=result_marker)
    
    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # Set title
    ax.set_title(title)
    
    # Display the plot
    plt.show()

def visualize_3d_trajectory(flight, trajectory, title='Circle Points', point_marker='o', point_color_range=(0, 1), 
                            trajectory_marker='o', trajectory_color='b', x_label='X', y_label='Y', z_label='Z'):
    """
    Visualize a 3D scatter plot, including original points and trajectory points.
    
    Parameters:
    flight: An object containing points and camera data
    trajectory: Coordinates of the trajectory points
    title: The title of the plot, default is 'Circle Points'
    point_marker: Marker style for spatial points, default is 'o'
    point_color_range: Range of colors for spatial points, default is (0, 1)
    trajectory_marker: Marker style for trajectory points, default is 'o'
    trajectory_color: Color for trajectory points, default is 'b' (blue)
    x_label: Label for the X-axis, default is 'X'
    y_label: Label for the Y-axis, default is 'Y'
    z_label: Label for the Z-axis, default is 'Z'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colors for original points
    colors = np.linspace(point_color_range[0], point_color_range[1], flight.points.shape[1])
    
    # Plot original spatial points
    ax.scatter(flight.points[0], flight.points[1], flight.points[2], c=colors, marker=point_marker, label='Original Points')
    
    # Plot trajectory points
    ax.scatter(trajectory[0], trajectory[1], trajectory[2], c=trajectory_color, marker=trajectory_marker, label='Trajectory Points')
    
    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # Set title
    ax.set_title(title)
    
    # Display legend
    ax.legend()
    
    # Display the plot
    plt.show()