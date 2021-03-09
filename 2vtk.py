#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import sys, os
import base64, zlib, glob

temp=[]

def main(model, start=1, end=-1):

    # Changing directory
    os.chdir(model)

    for i in range(start, end+1):
        nx = 512#len(x)
        nz = 128#len(z)

        print('Writing record #%03d' % i, end='\r')
        #sys.stdout.flush()
        fvts = open('f%03d.vts' % i, 'w')
        vts_header(fvts, nx, nz)

        # Read ffile, tfile, and wfile
        x,z,temp,alpha,mu = read_ffile(i)
        #tx,tz,tau,tauxx = read_tfile(i)
        chem = read_wfile(i)

        # Starting point data fields
        fvts.write('  <PointData>\n')
        # Temperature field
        temp = np.transpose(temp)
        vts_dataarray(fvts, temp, 'Temperature')
        #fvts.write('  </PointData>\n')

        # Fineness field
        #fvts.write('  <PointData>\n')
        alpha = np.transpose(alpha)
        vts_dataarray(fvts, alpha, 'Fineness')
        #fvts.write('  </PointData>\n')

        # Viscosity field
        #fvts.write('  <PointData>\n')
        mu = np.transpose(mu)
        vts_dataarray(fvts, mu, 'Viscosity')
        #fvts.write('  </PointData>\n')

        # Chemical field
        #fvts.write('  <PointData>\n')
        chem = np.transpose(chem)
        vts_dataarray(fvts, chem, 'Chemical')
        
        #Ending pointdata fields
        fvts.write('  </PointData>\n')

        # coordinate
        tmp = np.zeros((nx*nz,1,3),dtype=x.dtype)
        tmp[:,:,0] = x
        tmp[:,:,1] = z
        fvts.write('  <Points>\n')
        #vts_dataarray(fvts, tmp.swapaxes(0,1), '', 3)
        vts_dataarray(fvts, tmp, '', 3)
        fvts.write('  </Points>\n')


        vts_footer(fvts)
        fvts.close()
    print()
    return

def read_ffile(i):
    xraw,zraw,tempraw,alpharaw,muraw = np.loadtxt('f%03d' % i, usecols=[0, 1, 2, 3, 5], unpack=True)
    xc=xraw[0:-1:128] #x.reshape((len(x),1))
    zc=zraw[0:128]    #z.reshape((len(z),1))
    x = np.zeros((len(xraw),1))
    z = np.zeros((len(zraw),1))
    temp = np.zeros((len(tempraw),1))
    alpha = np.zeros((len(alpharaw),1))
    mu = np.zeros((len(muraw),1))
    for j in range(128):
        for i in range(512):
            norig = j + 128*i
            n = i + 512*j
            x[n,0] = xc[i]
            z[n,0] = zc[j]
            temp[n,0] = tempraw[norig]
            alpha[n,0] = alpharaw[norig]
            mu[n,0] = muraw[norig]
    return x,z,temp,alpha,mu

def read_tfile(i):
    tx,tz,tau,tauxx = np.loadtxt('t%03d' % i, usecols=[0, 1, 2, 3], unpack=True)
    tx=tx.reshape((len(tx),1))
    tz=tz.reshape((len(tz),1))
    tau=tau.reshape((len(tau),1))
    tauxx = tauxx.reshape((len(tauxx),1))
    return tx,tz,tau,tauxx 

def read_wfile(i):
    chem = np.loadtxt('w%03d' % i, usecols=3, unpack=True)
    chem = chem.reshape((len(chem),1))
    return chem

def vts_dataarray(f, data, data_name=None, data_comps=None):
    if data.dtype in (int, np.int32, np.int_):
        dtype = 'Int32'
    elif data.dtype in (float, np.single, np.double,
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
'''.format(nx-1, nz-1))
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
