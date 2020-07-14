#!/usr/bin/env python

import os
import shutil, glob

def make_link(src,dst):
    """ make_link(src,dst)
        makes a relative link
        Inputs:
            src - source file
            dst - destination to place link
    """
    
    assert os.path.exists(src) , 'source file does not exist \n%s' % src
    
    # find real file, incase source itself is a link
    src = os.path.realpath(src) 
    
    # normalize paths
    src = os.path.normpath(src)
    dst = os.path.normpath(dst)        

    # check for self referencing
    if src == dst: return        
    
    # find relative folder path
    srcfolder = os.path.join( os.path.split(src)[0] ) + '/'
    dstfolder = os.path.join( os.path.split(dst)[0] ) + '/'
    srcfolder = os.path.relpath(srcfolder,dstfolder)
    src = os.path.join( srcfolder, os.path.split(src)[1] )
    
    # make unix link
    if os.path.exists(dst): os.remove(dst)
    os.symlink(src,dst)
    

