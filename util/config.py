#!/usr/bin/env python

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os, sys, shutil, copy
import numpy as np
from ordered_dict import OrderedDict
from ordered_bunch import OrderedBunch
from switch import switch

inf = 1.0e20


# ----------------------------------------------------------------------
#  Configuration Class
# ----------------------------------------------------------------------

class Config(OrderedBunch):
    """ config = Config(filename="")
        
        Starts a config class, an extension of 
        ordered_bunch()
        
        use 1: initialize by reading config file
            config = Config('filename')
        use 2: initialize from dictionary or bunch
            config = Config(param_dict)
        use 3: initialize empty
            config = Config()
        
        Parameters can be accessed by item or attribute
        ie: config['MESH_FILENAME'] or config.MESH_FILENAME
        
        Methods:
            read()       - read from a config file
            write()      - write to a config file (requires existing file)
            dump()       - dump a raw config file
            unpack_dvs() - unpack a design vector 
            diff()       - returns the difference from another config
            dist()       - computes the distance from another config
    """    

    _filename = 'config.cfg'
    
    def __init__(self,*args,**kwarg):
        
        # look for filename in inputs
        if args and isinstance(args[0],str):
            filename = args[0]
            args = args[1:]
        elif 'filename' in kwarg:
            filename = kwarg['filename']
            del kwarg['filename']
        else:
            filename = ''
        
        # initialize ordered bunch
        super(Config,self).__init__(*args,**kwarg)
        
        # read config if it exists
        if filename:
            try:
                self.read(filename)
            except IOError:
                print('Could not find config file: %s' % filename)
            except:
                print('Unexpected error: ', sys.exc_info()[0])
                raise
        
        self._filename = filename
    
    def read(self,filename):
        """ reads from a config file """
        konfig = read_config(filename)
        self.update(konfig)
        
    def write(self,filename=''):
        """ updates an existing config file """
        if not filename: filename = self._filename
        assert os.path.exists(filename) , 'must write over an existing config file'
        write_config(filename,self)
        
    def dump(self,filename=''):
        """ dumps all items in the config bunch, without comments """
        if not filename: filename = self._filename
        dump_config(filename,self)
    
    def __getattr__(self,k):
        try:
            return super(Config,self).__getattr__(k)
        except AttributeError:
            raise AttributeError('Config parameter not found')
        
    def __getitem__(self,k):
        try:
            return super(Config,self).__getitem__(k)
        except KeyError:
            raise KeyError('Config parameter not found: %s' % k)

       
    def __eq__(self,konfig):
        return super(Config,self).__eq__(konfig)
    def __ne__(self,konfig):
        return super(Config,self).__ne__(konfig)
    
    
    def __repr__(self):
        #return '<Config> %s' % self._filename
        return self.__str__()
    
    def __str__(self):
        output = 'Config: %s' % self._filename
        for k,v in self.items():
            output +=  '\n    %s= %s' % (k,v)
        return output
#: class Config







# -------------------------------------------------------------------
#  Get Configuration Parameters
# -------------------------------------------------------------------

def read_config(filename):
    """ reads a config file """
      
    # initialize output dictionary
    data_dict = OrderedDict()
    
    input_file = open(filename)
    
    # process each line
    while 1:
        # read the line
        line = input_file.readline()
        if not line:
            break
        
        # remove line returns
        line = line.strip('\r\n')
        # make sure it has useful data
        if (not "=" in line) or (line[0] == '#'):
            continue
        # split across equals sign
        line = line.split("=",1)
        this_param = line[0].strip()
        this_value = line[1].strip()
        
        assert this_param not in data_dict, ('Config file has multiple specifications of %s' % this_param )
        for case in switch(this_param):
            
            # Put all INTEGER options here
            if case("np_init") or \
                case("np_braid") or \
                case("ntime") or \
                case("nspline") or \
                case("braid_maxlevels") or \
                case("braid_cfactor") or \
                case("braid_printlevel") or \
                case("braid_maxiter") or \
                case("braid_accesslevel") or \
                case("optim_maxiter") or \
                case("linearsolver_maxiter") or \
                case("output_frequency") or \
                case("optim_monitor_frequency") :
                data_dict[this_param] = int(this_value)
                break

            # Put all FLOAT options here
            if case("dt") or \
               case("braid_abstol") or \
               case("braid_reltol") or \
               case("optim_atol") or \
               case("optim_rtol") or \
               case("optim_penalty") or \
               case("optim_penalty_param") or \
               case("optim_regul") :
               data_dict[this_param] = float(this_value)
               break

            # Put all STRING option here
            if case("nlevels") or \
               case("nessential") or \
               case("selfkerr") or \
               case("crosskerr") or \
               case("Jkl") or \
               case("transfreq") or \
               case("rotfreq") or \
               case("carrier_frequency0") or \
               case("carrier_frequency1") or \
               case("carrier_frequency2") or \
               case("carrier_frequency3") or \
               case("carrier_frequency4") or \
               case("carrier_frequency5") or \
               case("collapse_type") or \
               case("decay_time") or \
               case("dephase_time") or \
               case("initialcondition") or \
               case("braid_fmg") or \
               case("braid_skip") or \
               case("optim_objective") or \
               case("optim_target") or \
               case("gate_rot_freq") or \
               case("optim_weights") or \
               case("optim_init") or \
               case("optim_init_ampl") or \
               case("optim_bounds") or \
               case("datadir") or \
               case("output0") or \
               case("output1") or \
               case("output2") or \
               case("output3") or \
               case("output4") or \
               case("output5") or \
               case("runtype") or \
               case("usematfree") or \
               case("linearsolver_type") or \
               case("apply_pipulse") :
               data_dict[this_param] = this_value
               break

    return data_dict
    
#: def read_config()


def dump_config(filename,config):
    ''' dumps a raw config file with all options in config 
        and no comments
    '''
           
    config_file = open(filename,'w')
    # write dummy file
    for key in config.keys():
        configline = str(key) + " = " + str(config[key]) + "\n"
        config_file.write(configline)
    config_file.close()
    # dump data
    #write_config(filename,config)    

