# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 10:13:16 2017

@author: Leonid Paramonov
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EllipticFit import elliptic_fit

if __name__ == "__main__" : 

    # This opens a handle to your file, in 'r' read mode
    file_handle = open( 'magRaw.dta', 'r' )
    
    # Read in all the lines of your file into a list of lines
    lines_list = file_handle.readlines()
    
    file_handle.close()
    
    # Do a double-nested list comprehension to get the rest of the data into your matrix
    raw_data = [ [float(val) for val in line.split()] for line in lines_list ]
    
    # numpy-fying the data matrix 
    raw_data_np = np.array( raw_data )
    
    print( 'raw data size: {}'.format( len(raw_data_np) ) )
    
    # caculating the fit
    ( center, radii, evecs, v ) = elliptic_fit( raw_data_np )
    print( 'Center: {0:.6f} {1:.6f} {2:.6f}'.format( center[0], center[1], center[2] ) )
    print( 'Radii:  {0:.6f} {1:.6f} {2:.6f}'.format( radii[0],  radii[1],  radii[2]  ) )
    print( 'Evecs:' )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( evecs[0,1], evecs[0,1], evecs[0,2] ) )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( evecs[1,1], evecs[1,1], evecs[1,2] ) )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( evecs[2,1], evecs[2,1], evecs[2,2] ) )

    # scaleMat = inv([radii(1) 0 0; 0 radii(2) 0; 0 0 radii(3)]) * min(radii);
    invMat = np.array([ \
                 np.array([ radii[0], 0., 0. ]), \
                 np.array([ 0., radii[1], 0. ]), \
                 np.array([ 0., 0., radii[2] ])  \
             ])

    scaleMat = np.linalg.inv( invMat ) * np.min( radii )
 
    # correctionMat = evecs * scaleMat * evecs';
    correctionMat = evecs.dot( scaleMat.dot( np.transpose( evecs ) ) )
    print( 'correctionMat:' )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( correctionMat[0,1], correctionMat[0,1], correctionMat[0,2] ) )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( correctionMat[1,1], correctionMat[1,1], correctionMat[1,2] ) )
    print( '{0:.6f} {1:.6f} {2:.6f}'.format( correctionMat[2,1], correctionMat[2,1], correctionMat[2,2] ) )

    # % take off center offset
    magVector = np.array( [ X-center for X in raw_data_np ] )
    
    # % do rotation and scale 
    # it is now transposed 
    magVector = correctionMat.dot( magVector.transpose() )
    
    x = raw_data_np[ :, 0 ]
    y = raw_data_np[ :, 1 ]
    z = raw_data_np[ :, 2 ]
    
    xCorr = magVector[ 0, : ]
    yCorr = magVector[ 1, : ]
    zCorr = magVector[ 2, : ]
    
    # plotting the data and the corrected data 
    dispRadius = 100

    fig = plt.figure()                  # the figure  
    ax = fig.gca( projection='3d' )     # 3D axis

    ax.set_title( 'Uncorrected samples (red) and corrected (blue) values' )
    
    ax.plot( x, y, z, 'r.' )
    ax.plot( xCorr, yCorr, zCorr, 'b.' )
    
    ax.set_aspect('equal')

    ax.set_xlim( [ -dispRadius, dispRadius ] )
    ax.set_ylim( [ -dispRadius, dispRadius ] )
    ax.set_zlim( [ -dispRadius, dispRadius ] )

    plt.show()


