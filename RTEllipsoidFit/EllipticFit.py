# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:44:59 2017

@author: Leonid Paramonov 

"""

import numpy as np

# elliptic fit function 
# copying the Matlab scripts structure 
def elliptic_fit( X ) :
    
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    # only rewriting the main branch of the Matlab fit function
    # without extra options ... easy to do later if needed

    # forming the D-matrix
    #     % fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    D = np.array([ \
        np.multiply( x, x ), \
        np.multiply( y, y ), \
        np.multiply( z, z ), \
        2.0 * np.multiply( x, y ), \
        2.0 * np.multiply( x, z ), \
        2.0 * np.multiply( y, z ), \
        2.0 * x, \
        2.0 * y, \
        2.0 * z \
        ])
    
    #    v = ( D' * D ) \ ( D' * ones( size( x, 1 ), 1 ) );
    DT = np.transpose( D )
    x_ones = np.transpose( np.ones( np.size(x) ) )
    v = np.linalg.solve( D.dot(DT), D.dot( x_ones ) )
    
    A = np.array([ \
        np.array([ v[0], v[3], v[4], v[6] ]), \
        np.array([ v[3], v[1], v[5], v[7] ]), \
        np.array([ v[4], v[5], v[2], v[8] ]), \
        np.array([ v[6], v[7], v[8], -1.0 ])  \
        ])
    
    center = np.linalg.solve( -A[ :3, :3 ], np.array([ v[6], v[7], v[8] ]) )
    
    T = np.eye( 4 )
    T[ 3, :3 ] = center

    R = T.dot( A.dot( np.transpose(T) ) )
    
    # % solve the eigenproblem
    # [ evecs evals ] = eig( R( 1:3, 1:3 ) / -R( 4, 4 ) );
    evals, evecs = np.linalg.eig( -R[ 0:3, 0:3 ] / R[ 3, 3 ] ) # signs of the eigvals!!!
    
    radii = np.divide( 1.0, np.sqrt( evals ) )    

    return center, radii, evecs, v  

# the main code 
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
    
    # caculating the fit
    ( center, radii, evecs, v ) = elliptic_fit( raw_data_np )

    # scaleMat = inv([radii(1) 0 0; 0 radii(2) 0; 0 0 radii(3)]) * min(radii);
    invMat = np.array([ \
                 np.array([ radii[0], 0., 0. ]), \
                 np.array([ 0., radii[1], 0. ]), \
                 np.array([ 0., 0., radii[2] ])  \
             ])

    scaleMat = np.linalg.inv( invMat ) * np.min( radii )
 
    # correctionMat = evecs * scaleMat * evecs';
    correctionMat = evecs.dot( scaleMat.dot( np.transpose( evecs ) ) )
    
    file_handle = open( 'magCorr.dta', 'w' )
    
    file_handle.write( \
        '{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f} {8:.6f} {9:.6f} {10:.6f} {11:.6f}\n'.format( \
        center[0], center[1], center[2], \
        correctionMat[0,0], correctionMat[0,1], correctionMat[0,2], \
        correctionMat[1,0], correctionMat[1,1], correctionMat[1,2], \
        correctionMat[2,0], correctionMat[2,1], correctionMat[2,2]  \
        ) )
    
