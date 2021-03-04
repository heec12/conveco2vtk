#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import sys, os
import base64, zlib, glob

temp=[]

def main(model, start=1, end=-1):

    # changing directory
    os.chdir(model)

    for i in range(start, end+1):
        nx = 512#len(x)
        nz = 128#len(z)

        print('Writing record #%03d' % i, end='\r')
        #sys.stdout.flush()
        fvts = open('f%03d.vts' % i, 'w')
        vts_header(fvts, nx, nz)

        # temperature field
        fvts.write('  <PointData>\n')

        a = read_temperature(i)
        a = np.transpose(a)
        vts_dataarray(fvts, a, 'Temperature')

        fvts.write('  </PointData>\n')

        # coordinate
        x, z = read_mesh(i)
        tmp = np.zeros((nx*nz,1,3),dtype=x.dtype)
        tmp[:,:,0] = x
        tmp[:,:,1] = z
        fvts.write('  <Points>\n')
        vts_dataarray(fvts, tmp.swapaxes(0,1), '', 3)
        fvts.write('  </Points>\n')


        vts_footer(fvts)
        fvts.close()
    print()
    return

def read_temperature(i):
    temp = np.loadtxt('f%03d' % i, usecols=2, unpack=True)
    temp=temp.reshape((len(temp),1))
    #temp = np.fromfile('f%03d' % i, sep=' ')
    return temp

def read_mesh(i):
    x,z = np.loadtxt('f%03d' % i, usecols=[0, 1], unpack=True)
    x=x.reshape((len(x),1))
    z=z.reshape((len(z),1))
    return x,z

def vts_dataarray(f, data, data_name=None, data_comps=None):
    if data.dtype in (int, np.int32, np.int_):
        dtype = 'Int32'
    elif data.dtype in (float, np.single, np.double, np.float,
                        np.float32, np.float64, np.float128):
        dtype = 'Float32'
    else:
        print('Unknown data type!!')

    name = ''
    if data_name:
        name = 'Name="{0}"'.format(data_name)

    ncomp = ''
    if data_comps:
        ncomp = 'NumberOfComponents="{0}"'.format(data_comps)

    fmt = 'ascii'
    header = '<DataArray type="{0}" {1} {2} format="{3}">\n'.format(
        dtype, name, ncomp, fmt)
    f.write(header)

    data.tofile(f, sep=' ')
    f.write('\n</DataArray>\n')
    return

def vts_header(f, nx, nz):
    f.write(
'''<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">
<StructuredGrid WholeExtent="0 {0} 0 {1} 0 0">
    <Piece Extent="0 {0} 0 {1} 0 0">
'''.format(nx, nz))
    return

def vts_footer(f):
    f.write(
'''</Piece>
</StructuredGrid>
</VTKFile>
''')
    return

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('''usage: 2vtk.py model [step_min [step_max]]
Processing f data output to VTK format.
If step_max is not given, processing to latest steps
If both step_min and step_max are not given, processing all steps''')
        sys.exit(1)

    model = sys.argv[1]

    start = 1
    end = -1
    if len(sys.argv) >= 3:
        start = int(sys.argv[2])
        if len(sys.argv) >= 4:
            end = int(sys.argv[3])

    main(model, start, end)