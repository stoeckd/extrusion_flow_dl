#!/usr/bin/env python 
# image 2 gmsh
import os
# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
t_start = time.perf_counter()
import gmsh
import numpy as np
from scipy.interpolate import griddata
from PIL import Image

from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
import math
from rdp import rdp

# fenicsx solver
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

import sys

# Visualization
import pyvista
from dolfinx.plot import vtk_mesh

# Inlet processing
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.vectorized import contains

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
tf.keras.config.enable_unsafe_deserialization()

t_stop = time.perf_counter()
#print(f'Imports complete in {t_stop - t_start:.4f} seconds')

def process_image_channels(img):
    if (len(img.shape) == 2):
        gray_img = img
    else:
        if (img.shape[2] == 3):
            gray_img = sk.color.rgb2gray(img)
        if (img.shape[2] == 4):
            rgb_img = sk.color.rgba2rgb(img)
            gray_img = sk.color.rgb2gray(rgb_img)

    return gray_img

def load_image(img_fname):
    ''' 
    Load image as a grayscale image
    '''
    #print('Loading image {}'.format(img_fname))
    img = sk.io.imread(img_fname)

    gray_img = process_image_channels(img)

    return gray_img

def get_contours(gray_img):
    '''
    Extract contours from image for mesh generation
    '''
    height, width = gray_img.shape    
    # Normalize and flip (for some reason)
    raw_contours = sk.measure.find_contours(gray_img, 0.5)
 
    #print('Found {} contours'.format(len(raw_contours)))

    contours = []
    for n, contour in enumerate(raw_contours):
        # Create an empty image to store the masked array
        r_mask = np.zeros_like(gray_img, dtype = int)  # original np.int
        # Create a contour image by using the contour coordinates 
        # rounded to their nearest integer value
        r_mask[np.round(contour[:, 0]).astype('int'), 
               np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        r_mask = ndimage.binary_fill_holes(r_mask)

        contour_area = float(np.count_nonzero(r_mask))/(float(height * width))
        #print(np.count_nonzero(r_mask))
        if (contour_area >= 0.05):
            contours.append(contour)

    #print('Reduced to {} contours'.format(len(contours)))

    for n, contour in enumerate(contours):
        contour[:,1] -= 0.5 * height
        contour[:,1] /= height

        contour[:,0] -= 0.5 * width
        contour[:,0] /= width
        # contour[:,0] *= -1.0

    print("{:d} Contours detected in geometry".format(len(contours)))

    return contours


def optimize_contour(contour):
    '''
    Reduce number of points in contour but maintain overall shape
    '''
    #print("Optimizing contour.")
    dir_flag = 0
    dir_bank = []

    contour_keep = []

    ## Use low-pass fft to smooth out 
    x = contour[:,1]
    y = contour[:,0]

    signal = x + 1j*y
    #print(signal)

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    cutoff = 0.12
    fft[np.abs(freq) > cutoff] = 0 

    signal_filt = np.fft.ifft(fft)

    contour[:,1] = signal_filt.real
    contour[:,0] = signal_filt.imag


    #contour = rdp(contour)
    contour = rdp(contour, epsilon=0.0005)

    # Remove final point in RDP, which coincides with
    # the first point
    contour = np.delete(contour, len(contour)-1, 0)

    # cutoff of 0.15, eps of 0.005 works for inner flow

    #contour = reverse_opt_pass(contour)
    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])
    
    max_y = max(contour[:,0])
    min_y = min(contour[:,0])
    
    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    mesh_lc = 0.01 * lc    

    return [contour, mesh_lc]

def outer_contour_to_gmsh(contour, mesh_lc, p_idx=1, l_idx=1, loop_idx=1):
    '''
    Converge outer wall contour to mesh for outer flow
    '''
    #print('Running outer_contour_to_gmsh')
    line_init = l_idx
    g = gmsh.model.geo
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("outer_contour_mesh")
    lc = mesh_lc
    #lc = 0.1 # comment out this line
    g.addPoint(-0.5, -0.5, 0, lc, p_idx)
    g.addPoint( 0.5, -0.5, 0, lc, p_idx + 1)
    g.addPoint( 0.5,  0.5, 0, lc, p_idx + 2)
    g.addPoint(-0.5,  0.5, 0, lc, p_idx + 3)
    
    g.addLine(p_idx, p_idx + 1, l_idx)
    g.addLine(p_idx + 1, p_idx + 2, l_idx + 1)
    g.addLine(p_idx + 2, p_idx + 3, l_idx + 2)
    g.addLine(p_idx + 3, p_idx, l_idx + 3)
    g.addCurveLoop(list(range(l_idx , l_idx + 4)), loop_idx)
    
    p_idx += 3
    p_idx_closure = p_idx + 1
    for point in contour:
        p_idx += 1
        g.addPoint(point[1], point[0], 0, lc, p_idx)
    
    l_idx += 3
    for line in range(len(contour)-1):
        l_idx += 1
        g.addLine(l_idx, l_idx + 1, l_idx)
        
    g.addLine(l_idx + 1, p_idx_closure, l_idx + 1)
    g.addCurveLoop(list(range(p_idx_closure , l_idx + 2)), 2)
    g.addPlaneSurface([1,2], 1)
    g.synchronize()
    
    gmsh.model.addPhysicalGroup(1, list(range(1 , l_idx + 2)), name = "walls")
    gmsh.model.addPhysicalGroup(2, [1], name = "outer_surface") 
    
    gmsh.model.mesh.generate(2)
    # gmsh.write("outer_contour_mesh.msh")
 
    return gmsh.model



def inner_contour_to_gmsh(contour, mesh_lc):
    '''
    Converge inner wall contour to mesh for outer flow
    '''
    #print('Running inner_contour_to_gmsh')
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("inner_contour_mesh")
    g = gmsh.model.geo
    lc = mesh_lc
    #lc = 6  # comment out this line
    idx = 0
    for point in contour:
        idx += 1
        g.addPoint(point[1], point[0], 0, lc, idx)
    idx = 0
    for line in range(len(contour)-1):
        idx += 1
        g.addLine(idx, idx + 1, idx)
    g.addLine(idx + 1, 1, idx + 1)
    g.addCurveLoop(list(range(1 , idx + 2)), 1)
    g.addPlaneSurface([1], 1)
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, list(range(1 , idx + 2)), name = "walls")
    gmsh.model.addPhysicalGroup(2, [1], name = "inner_surface") 
    gmsh.model.mesh.generate(2)

    # gmsh.write("inner_contour_mesh.msh")
    
    return gmsh.model

def process_2_channel_mesh_model(contours):
    '''
    Create inner and outer meshes for inlet flow profiles
    '''
    contour_inner = contours[1]
    contour_outer = contours[0]

    [contour_inner, mesh_lc_a] = optimize_contour(contour_inner)
    [contour_outer, mesh_lc_c] = optimize_contour(contour_outer)

    inner_shape = create_inner_shape2(contour_inner)

    inner_model = inner_contour_to_gmsh(contour_inner, mesh_lc_a)
    outer_model = outer_contour_to_gmsh(contour_outer, mesh_lc_c)

    return inner_model, outer_model, inner_shape

def image2gmshfromimg(img):
    '''
    Convert inlet nozzle geometry image to gmsh models and inner flow shape
    '''
    contours = get_contours(img)
    inner_model, outer_model, inner_shape = process_2_channel_mesh_model(contours)

    return inner_model, outer_model, inner_shape

def image2gmshfromfile(img_fname):
    '''
    Convert inlet nozzle geometry image to gmsh models and inner flow shape
    '''
    img = load_image(img_fname)

    inner_model, outer_model, inner_shape = image2gmshfromimg(img)

    return inner_model, outer_model, inner_shape

def solve_velocity_field(gmsh_model):
    '''
    Given a gmsh model for an inlet flow field, solve diffusion equation
    to model a fully developed inlet flow field
    '''
    gdim = 2
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh_model, mesh_comm, gmsh_model_rank, gdim=gdim)

    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Defining an arbitrary spatially varying load
    # This can be any positive number, as the velocity field will be
    # normalized anyway.
    p = 10
    
    # Find Area
    one = fem.Constant(domain, PETSc.ScalarType(1))
    f = fem.form(one*ufl.dx)
    area = fem.assemble_scalar(f)

    noslip = fem.Constant(domain, PETSc.ScalarType(0))
    dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, facet_markers.find(1))
    bc = fem.dirichletbc(noslip, dofs, V)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = p * v * ufl.dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    f2 = fem.form(uh*ufl.dx)
    average_velocity = fem.assemble_scalar(f2)/area

    
    # topology, cell_types, x = vtk_mesh(domain, domain.topology.dim)
    # grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    # grid.point_data["u"] = uh.x.array
    # warped = grid.warp_by_scalar("u", factor= 0.25 )
    # plotter = pyvista.Plotter()
    # plotter.background_color = "white"
    # plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False, scalars="u")
    # #plotter.show_bounds()
    # if not pyvista.OFF_SCREEN:
    #     plotter.show()
    # else:
    #     plotter.screenshot("deflection.png") 

    return uh, area, average_velocity, domain, V

def solve_inlet_profiles(inner_model, outer_model, inner_shape, flowrate_ratio, normalize_factor=1.0):
    '''
    Given an image of nozzle geometry and a flowrate ratio,
    solve both inner and outer flow profiles
    '''
    t_start = time.perf_counter()
    inner_model.setCurrent("inner_contour_mesh")
    uh_1, area_1, average_u_1, domain_1, V_1 = solve_velocity_field(inner_model)
    outer_model.setCurrent("outer_contour_mesh")
    uh_2, area_2, average_u_2, domain_2, V_2 = solve_velocity_field(outer_model)
    t_stop = time.perf_counter()

    flow_solve_time = t_stop - t_start
    print(f'Flow fields solved in {flow_solve_time:.4f} seconds')

    t_start = time.perf_counter()

    uh_1.x.array[:] = uh_1.x.array[:] / average_u_1
    uh_2.x.array[:] = uh_2.x.array[:] / average_u_2

    f1 = fem.form(uh_1*ufl.dx)
    average_velocity_1 = fem.assemble_scalar(f1)/area_1
    f2 = fem.form(uh_2*ufl.dx)
    average_velocity_2 = fem.assemble_scalar(f2)/area_2

    flow_u_1 = flowrate_ratio / area_1
    flow_u_2 = (1.0 - flowrate_ratio) / area_2

    uh_1.x.array[:] = uh_1.x.array[:] * flow_u_1
    uh_2.x.array[:] = uh_2.x.array[:] * flow_u_2

    f1 = fem.form(uh_1*ufl.dx)
    average_velocity_1 = fem.assemble_scalar(f1)/area_1
    f2 = fem.form(uh_2*ufl.dx)
    average_velocity_2 = fem.assemble_scalar(f2)/area_2

    #print(f'Updated average_1 = {average_velocity_1}') 
    #print(f'Updated average_2 = {average_velocity_2}') 

    coor_1 = V_1.tabulate_dof_coordinates()
    u_1 = uh_1.x.array.real.astype(np.float32)
    coor_2 = V_2.tabulate_dof_coordinates()
    u_2 = uh_2.x.array.real.astype(np.float32)

    coordinates = np.vstack((coor_1, coor_2))
    vertex_values = np.concatenate((u_1, u_2))

    t_stop = time.perf_counter()
    adjust_time = t_stop - t_start
    #print(f'Flow fields adjusted in {adjust_time:.4f} seconds')
    
    # Interpolate velocity onto zero domain
    t_start = time.perf_counter()
    x = np.linspace(-0.5, 0.5, 256)
    y = np.linspace(-0.5, 0.5, 256)
    grid_x, grid_y = np.meshgrid(x,y)
    #vertex_values = np.uint8(vertex_values/np.max(vertex_values) * 255)
    grid = griddata(coordinates[:,:2], vertex_values, (grid_x, grid_y), method='linear', fill_value=0.0)

    #np.savetxt('fenicsx_grid.txt', grid)

    local_max_u = np.amax(grid)
    #print(f'Local normalizing factor = {local_max_u}')

    if (local_max_u > normalize_factor):
        #print(f'Using local normalizing factor = {local_max_u}')
        #print(f'Standard normalizing factor = {normalize_factor}')
        normalize_factor = local_max_u

    grid /= normalize_factor
    grid *= 255
    flow_profile = grid.astype(np.uint8)

    t_stop = time.perf_counter()
    interpolate_time = t_stop - t_start
    print(f'Flow fields combined in {interpolate_time:.4f} seconds')

    #img = Image.fromarray(flow_profile, 'L')
    #img.save('inlet_flow_profile_fenicsx.png')

    return flow_profile, inner_shape

def load_normalize_factor():
    '''
    Need to know the normalizing factor for all velocity fields
    basd on the training dataset distribution
    '''
    fname = 'u_max.npz'
    lib_fname = (
        os.path.join(os.path.dirname(sys.executable), "share", fname)
        if getattr(sys, "frozen", False)
        else os.path.join(os.path.dirname(__file__), fname)
        )
    u_max = np.load(fname)['arr_0'].item()
    return u_max

def create_inner_shape(contour_points):
    '''
    Form inlet inner flow shape based on inner contour
    '''
    t_start = time.perf_counter()
    polygon = Polygon(contour_points)

    nx = 256
    ny = 256

    y_min = -0.5
    y_max = 0.5
    x_min = -0.5
    x_max = 0.5

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    grid = np.zeros((nx,ny), dtype=np.uint8)

    #f = open(shape_filename, 'w')
    for j in range(0, ny):
        for i in range(0, nx):
            point = Point(x[i], y[j])

            if polygon.contains(point):
                grid[i,j] = 255
                #f.write('1\n')
            else:
                grid[i,j] = 0
                #f.write('0\n')

    #f.close()
    # inner_img = Image.fromarray(grid, 'L')
    # img.save('inner_shape.png')
    t_stop = time.perf_counter()
    inner_shape_time = t_stop - t_start
    print(f'Inner shape created in {inner_shape_time:.4f} seconds')

    return grid

def create_inner_shape2(contour_points):
    '''
    Form inlet inner flow shape based on inner contour
    '''
    t_start = time.perf_counter()
    polygon = Polygon(contour_points)

    nx = 256
    ny = 256

    y_min = -0.5
    y_max = 0.5
    x_min = -0.5
    x_max = 0.5

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    xx, yy = np.meshgrid(x, y)

    # Flatten the grid for efficient processing
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # Vectorized containment check
    #mask = np.array([polygon.contains(Point(x, y)) for x, y in grid_points])
    inner_shape_mask = 255 * contains(polygon, xx, yy).T

    '''
    img = convert_binary_to_color(grid)
    img = Image.fromarray(img, 'RGB')
    img.save('inner_shape.png')
    '''

    t_stop = time.perf_counter()
    inner_shape_time = t_stop - t_start

    print(f'Inner shape created in {inner_shape_time:.4f} seconds')

    return inner_shape_mask

def load_flownet_model():
    '''
    Locate and load flownet model
    '''
    t_start = time.perf_counter()
    fname = 'flownet_model.keras'
    lib_fname = (
        os.path.join(os.path.dirname(sys.executable), "share", fname)
        if getattr(sys, "frozen", False)
        else os.path.join(os.path.dirname(__file__), fname)
        )

    flownet_model = load_model(lib_fname)

    t_stop = time.perf_counter()
    flownet_model_load_time = t_stop - t_start
    #print(f'Flownet model loaded in {flownet_model_load_time:.4} seconds')

    return flownet_model

def create_2ch_test_data_from_img(ch1_img, ch2_img, h, w):
    '''
    Convert inlet velocity field and inner shape into tensorflow-compatible
    2-channel image
    '''
    X_test = np.zeros((1, h, w, 2), dtype=np.uint8)

    ch1_img = ch1_img.reshape((ch1_img.shape[0], ch1_img.shape[1], 1))
    ch1_img = sk.transform.resize(ch1_img, (h, w), mode='constant', preserve_range=True)

    X_test[0,:,:,0] = ch1_img.squeeze()

    ch2_img = ch2_img.reshape((ch2_img.shape[0], ch2_img.shape[1], 1))
    ch2_img = sk.transform.resize(ch2_img, (h, w), mode='constant', preserve_range=True)

    X_test[0,:,:,1] = ch2_img.squeeze()
    
    return X_test

def run_flownet_cpu(X_test):
    '''
    Predict outlet flow using CPU, loading the model each time
    '''

    with tf.device('/cpu:0'):
        flownet_model = load_flownet_model()

        pred_test = flownet_model.predict(X_test, verbose=0)
        pred_test_t = (pred_test > 0.5).astype(np.uint8)
        pred_mask = pred_test_t[0].squeeze()
        
    return pred_mask

def run_flownet_cpu_preload_model(flownet_model, X_test):
    '''
    Given a pre-loaded flownet model, predict outlet flow
    '''

    with tf.device('/cpu:0'):
        pred_test = flownet_model.predict(X_test, verbose=0)
        pred_test_t = (pred_test > 0.5).astype(np.uint8)
        pred_mask = pred_test_t[0].squeeze()
        
    return pred_mask

def convert_binary_to_color(img_array):
    '''
    Convert black/white or yellow/purple image to blue/white to match
    publication colors
    '''
    w,h = img_array.shape

    img_array = img_array.astype(np.uint8)

    zero_loc = np.where(img_array == 0)
    color_loc = np.where(img_array == img_array.max())

    img_array[zero_loc] = 255
    img_array[color_loc] = 0.0

    img_r = img_array.copy()
    img_g = img_array.copy()
    img_b = img_array.copy()

    img_r[color_loc] = 81
    img_g[color_loc] = 164
    img_b[color_loc] = 209

    img_out = np.zeros((w,h,3), dtype=np.uint8)
    img_out[:,:,0] = img_r
    img_out[:,:,1] = img_g
    img_out[:,:,2] = img_b

    return img_out

def run_job_preload_model(flownet_model, img_fname, flowrate_ratio, img_pred_fname):
    '''
    Complete job function, given a pre-loaded flownet model
    '''
    u_max = load_normalize_factor()
    
    t_start = time.perf_counter()
    inner_model, outer_model, inner_shape = image2gmshfromfile(img_fname)
    t_stop = time.perf_counter()
    gmsh_time = t_stop - t_start
    print(f'Meshing complete in {gmsh_time:.4f} seconds')

    flow_profile, inner_shape = solve_inlet_profiles(inner_model, outer_model, inner_shape, flowrate_ratio, u_max)    

    t_start = time.perf_counter()
    X_test = create_2ch_test_data_from_img(flow_profile, inner_shape, 256, 256)
    t_stop = time.perf_counter()
    test_data_creation_time = t_stop - t_start
    #print(f'Test data created in {test_data_creation_time:.4f} seconds')


    t_start = time.perf_counter()
    pred_mask = run_flownet_cpu_preload_model(flownet_model, X_test)
    t_stop = time.perf_counter()
    pred_time = t_stop - t_start
    print(f'Flownet prediction in {pred_time:.4f} seconds')

    img_pred_mask = convert_binary_to_color(pred_mask)
    img_pred_mask = Image.fromarray(img_pred_mask, 'RGB')
    #img_pred_mask = Image.fromarray(pred_mask*255)
    #img_pred_mask = img_pred_mask.convert("L")
    img_pred_mask.save(img_pred_fname)


def run_job_preload_model_preload_img(flownet_model, img, flowrate_ratio, img_pred_fname):
    '''
    Complete job function, given a pre-loaded flownet model
    '''
    print('-------------------')
    u_max = load_normalize_factor()
    t_start = time.perf_counter()
    inner_model, outer_model, inner_shape = image2gmshfromimg(img)
    t_stop = time.perf_counter()
    gmsh_time = t_stop - t_start
    print(f'Meshing complete in {gmsh_time:.4f} seconds')

    flow_profile, inner_shape = solve_inlet_profiles(inner_model, outer_model, inner_shape, flowrate_ratio, u_max)    

    t_start = time.perf_counter()
    X_test = create_2ch_test_data_from_img(flow_profile, inner_shape, 256, 256)
    t_stop = time.perf_counter()
    test_data_creation_time = t_stop - t_start
    #print(f'Test data created in {test_data_creation_time:.4f} seconds')


    t_start = time.perf_counter()
    pred_mask = run_flownet_cpu_preload_model(flownet_model, X_test)
    t_stop = time.perf_counter()
    pred_time = t_stop - t_start
    print(f'Flownet prediction in {pred_time:.4f} seconds')

    img_pred_mask = convert_binary_to_color(pred_mask)
    img_pred_mask = Image.fromarray(img_pred_mask, 'RGB')
    #img_pred_mask = Image.fromarray(pred_mask*255)
    #img_pred_mask = img_pred_mask.convert("L")
    #img_pred_mask.save(img_pred_fname)

    return img_pred_mask

def run_job(img_fname, flowrate_ratio, img_pred_fname):
    '''
    Complete job function, but loads flownet each time
    '''
    u_max = load_normalize_factor()

    t_start = time.perf_counter()
    inner_model, outer_model, inner_shape = image2gmshfromfile(img_fname)
    t_stop = time.perf_counter()
    gmsh_time = t_stop - t_start

    print(f'Meshing complete in {gmsh_time:.4f} seconds')

    flow_profile, inner_shape = solve_inlet_profiles(inner_model, outer_model, inner_shape, flowrate_ratio, u_max)    

    t_start = time.perf_counter()
    X_test = create_2ch_test_data_from_img(flow_profile, inner_shape, 256, 256)
    t_stop = time.perf_counter()
    test_data_creation_time = t_stop - t_start
    print(f'Test data created in {test_data_creation_time:.4f} seconds')


    t_start = time.perf_counter()
    pred_mask = run_flownet_cpu(X_test)
    t_stop = time.perf_counter()
    pred_time = t_stop - t_start
    print(f'Flownet prediction in {pred_time:.4f} seconds')

    img_pred_mask = convert_binary_to_color(pred_mask)
    img_pred_mask = Image.fromarray(img_pred_mask, 'RGB')
    #img_pred_mask = Image.fromarray(pred_mask*255)
    #img_pred_mask = img_pred_mask.convert("L")
    img_pred_mask.save(img_pred_fname)

    return img_pred_mask

#def flownet_predict(img


if __name__ == '__main__':
    img_fname = sys.argv[1]
    flowrate_ratio = float(sys.argv[2])
    img_pred_fname = sys.argv[3]

    img = load_image(img_fname)

    flownet_model = load_flownet_model()

    #run_job(img_fname, flowrate_ratio, img_pred_fname)

    img_out = run_job_preload_model_preload_img(flownet_model, img, flowrate_ratio, img_pred_fname)




