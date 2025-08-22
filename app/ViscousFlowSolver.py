#!/usr/bin/env python

import numpy as np
import sys
import os
from mpi4py import MPI

from streamtrace import for_and_rev_streamtrace
from NavierStokesChannelFlow import solve_NS_flow, make_output_folder, write_run_metadata, save_navier_stokes_solution
from dolfinx import geometry

from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, Function

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def CFD_solver_and_streamtrace(Reynolds_number, img_fname, channel_mesh_size, flow_ratio, num_seeds):
    print("Flow ratio in CFD_solver_and_streamtrace:")
    print(flow_ratio)

    msh, uh, uvw_data, xyz_data, Re, img_fname, channel_mesh_size, V, Q, flow_ratio, u, p = solve_NS_flow(
            Reynolds_number, img_fname, channel_mesh_size, flow_ratio)

    np.set_printoptions(threshold=100)  # or threshold=sys.maxsize
    limits = 1

    u.x.scatter_forward()
    p.x.scatter_forward()

    # Interpolate on all ranks (parallel)
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    p_out = Function(functionspace(msh, P3))
    p_out.interpolate(p)

    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u_out = Function(functionspace(msh, P4))
    u_out.interpolate(u)

    # Open XDMF files in parallel (using comm, not MPI.COMM_SELF)
    with XDMFFile(comm, f"Re{Re}ChannelPressure.xdmf", "w") as pfile_xdmf:
        p_out.name = "Pressure"
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p_out)

    with XDMFFile(comm, f"Re{Re}ChannelVelocity.xdmf", "w") as ufile_xdmf:
        u_out.name = "Velocity"
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u_out)

    rev_streamtrace_fig = for_and_rev_streamtrace(
        num_seeds, limits, img_fname, msh, u, uvw_data, xyz_data, Re)

    return rev_streamtrace_fig