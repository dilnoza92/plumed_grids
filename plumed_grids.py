#
# plumed_grids.py
#
# Written by Andrew White
# See LICENSE and README file included in repo
#
from math import ceil, floor, log, exp, pi, copysign
import numpy as np
from scipy.integrate import simps
import copy

NDITER = True
try:
    np.nditer
except AttributeError:
    NDITER = False



class Grid(object):
    """PLUMED grid class
    
    Member variables: 
      grid_points: number of points in the pot for each cv
      min: min for each cv
      max: max for each cv
      periodic: logical, indicating periodicity for each cv
      ncv: number of cv
      dx: grid spacing for each cv
      pot: array for potential      
      types: type for each cv
    """
    cv_type_map = {"Distance":1,
                    "Minimum distance":2,
                    "Coordination number":3,
                    "Angle":4,
                    "Torsion":5,
                    "Alpha-beta similarity":6,
                    "Hydrogen bonds":7,
                    "Dipole":8,
                    "Radius of gyration":11,
                    "Dihedral correlation":16,
                    "Interfacial water":20,
                    "Path collective variable S":30,
                    "Path collective variable Z":31,
                    "Absolute position":32,
                    "Electrostatic potential":33,
                    "Puckering coordinates":34,
                    "Energy":35,
                    "Helix loops":36,
                    "Alpha helix rmsd":37,
                    "Antiparallel beta rmsd":38,
                    "Parallel beta rmsd":39,
                    "PCA projection":42,
                    "Contact Map":45,
                    "SPRINT":55}

    def __init__(self):
        """Create a grid. Add individual CVs using the add_cv method"""
        self._clear()

    def _clear(self):
        """Clear everything in in grid"""
        self.min = []
        self.max = []
        self.types = []
        self.periodic = []
        self.pot = None
        self.meshgrid = None
        self._grid_points = None


    def clone(self):
        """Make a deep copy of the grid
        """
        g = Grid()
        g.min = copy.copy(self.min)
        g.max = copy.copy(self.max)
        g.types = copy.copy(self.types)
        g.periodic = copy.copy(self.periodic)
        g.pot = np.copy(self.pot)
        g._grid_points = self._grid_points
        return g

    def load_data(self, filename, reset_bounds=True):
        """Read data line by line from a file and load it into the potential

        The data should specify a grid points and the potential in
        that bin/grid point. If a dimension is periodic, it's
        assumed the final grid point wraps to the first point."""

        data = np.genfromtxt(filename)
        if(np.shape(data)[1] != self.dims + 1):
            raise ValueError("Incorrect number of dimensions in file {}".format(filename))

        #find number of unique in each dimension
        uniques = [np.unique(data[:,i]) for i in range(self.dims)]
        dx = []
        for u in uniques:
            dx.append((np.max(u) - np.min(u)) / len(u))

        if(reset_bounds):
            for x in data:
                self.min = [min(m,xi) for xi,m in zip(x[:-1], self.min)]
                self.max = [max(m,xi) for xi,m,dxi in zip(x[:-1], self.max,dx)]        
        

        
        old_grid_points = self.grid_points
        self.set_grid_point_number( [len(np.unique(data[:,i])) for i in range(self.dims)] ) 
        self.pot[:] = 0
        for x in data:
            indexs = self.np_to_index(x[:-1])

            assert self.pot[tuple(indexs)] == x[-1] or self.pot[tuple(indexs)] == 0, "Error, value {} was set to 2 different values {} and {}, likely due to periodic bounddary wrapping".format(x[:-1], x[-1], self.pot[tuple(indexs)])

            self.pot[tuple(indexs)] = x[-1]
        self.set_grid_point_number(old_grid_points)

    def read_plumed_grid(self, filename):
        '''
        Guesses format (plumed1/plumed2) and dispatches to correct method
        '''

        if(type(filename) != type('')):
            raise ValueError('Please pass a filename (string)')

        plumed1 = False
        with open(filename, 'r') as f:
            line = f.readline()
            while(line.find('#!') == -1):
                line = f.readline()
            if(line.find('FIELDS') == -1):
                plumed1 = True
        if(plumed1):
            self.read_plumed1_grid(filename)
        else:
            self.read_plumed2_grid(filename)
                    
            
        


    def read_plumed1_grid(self, filename):

        import re
        self._clear()
        
        #I'll ignore the force for now
        with open(filename, 'r') as f:
            line = f.readline()
            while(line.find('#!') != -1):
                if(line.find('TYPE') != -1):
                    self.types = [int(x) for x in re.findall(r'\d{1,}', line)]
                if(line.find('MIN') != -1):
                    self.min = [float(x) for x in re.findall(r'-*\d+\.*\d*', line)]
                if(line.find('MAX') != -1):
                    self.max = [float(x) for x in re.findall(r'-*\d+\.*\d*', line)]
                if(line.find('BIN') != -1):
                    bins = [int(x) for x in re.findall(r'\d{1,}', line)]
                if(line.find('NVAR') != -1):
                    ncv = [int(x) for x in re.findall(r'\d{1,}', line)]
                    ncv = ncv[0]
                if(line.find('PBC') != -1):
                    self.periodic = [int(x) == 1 for x in re.findall(r'\d{1,}', line)]
                line = f.readline()

        #convert to grid_points
        grid_points = bins
        for i in range(len(grid_points)):
            if(not self.periodic[i]):
                grid_points[i] += 1

        #now load data
        data = np.genfromtxt(filename)        
        #check header
        assert np.shape(data)[0] == reduce(lambda x,y: x * y, grid_points, 1), "Number of lines in grid does not match stated grid point number (calculated from NBINS): read {}, grid points = {} => {}".format(np.shape(data)[0], grid_points, reduce(lambda x,y: x * y, grid_points, 1))


        #build the grid
        self.pot = data[:,ncv]
        #switch from fortran order
        self.pot = np.reshape(self.pot, grid_points, order='F')

    def read_plumed2_grid(self, filename):
        
        import re

        #define a helper function to parse header
        def get_attr(names, attr, line):
            set_name = re.match('.*{}_(\S*)\s*(\S*)$'.format(attr), line)
            if(set_name is not None and len(set_name.groups()) == 2):
                i = names.index(set_name.group(1))
                if(i != -1):
                    value = set_name.group(2)
                    if(value == 'pi'):
                        value = pi
                    elif(value == '-pi'):
                        value = -pi
                    return i, value
            return -1, None

            
        
        self._clear()
        
        #I'll ignore the force for now
        with open(filename, 'r') as f:

            toset = 9999
            toskip = 0
            
            while True:                
                line = f.readline()
                if line.find('#!') == -1:
                    break
                if(line.find('FIELDS') != -1):
                    #assume first two tokens are #! and FIELDS
                    self.names = []                    

                    #Skip derivatives and variables
                    for item in line.split()[2:]:
                        if(item.startswith('der_')):
                            toskip += 1
                        else:
                            assert toskip == 0, "Order of fields is unexpected"
                            self.names.append(item)
                    #pop the last name, which is the name of the variable being measured
                    print 'Reading in variable called {}'.format(self.names.pop())
                    #self.names.pop()
                    ncv = len(self.names)
                    toset = ncv * 4
                    bins = [0 for i in xrange(ncv)]
                    self.min = [0 for i in xrange(ncv)]
                    self.max = [0 for i in xrange(ncv)]
                    self.periodic = [0 for i in xrange(ncv)]
                    self.types = [0 for i in xrange(ncv)]
                    continue
                assert hasattr(self, 'names'), 'Was not able to parse names from plumed2 grid {}'.format(f.name)
                #set attributes
                i,value = get_attr(self.names, 'min', line)
                if(i != -1):
                    self.min[i] = float(value)
                    toset -= 1
                    continue
                
                i,value = get_attr(self.names, 'max', line)
                if(i != -1):
                    self.max[i] = float(value)
                    toset -= 1
                    continue
                
                i,value = get_attr(self.names, 'periodic', line)
                if(i != -1):
                    if(value == 'true'):
                        self.periodic[i] = True
                    else:
                        self.periodic[i] = False
                    toset -= 1
                    continue

                i,value = get_attr(self.names, 'nbins', line)
                if(i != -1):
                    bins[i] = int(value)
                    toset -= 1
                    continue
                
                print 'WARNING: Did not know what do with this line:\n', line

            if toset != 0:
                raise IOError('Could not determine all attributes from header. {} Remained unset'.format(toset))
                                                        
        #convert to grid_points
        grid_points = bins
        for i in range(len(grid_points)):
            if(not self.periodic[i]):
                grid_points[i] += 1
        
        #now load data
        data = np.genfromtxt(filename)
        assert data.shape[1] == ncv + toskip + 1, "Invalid number of columns. Should have {} for CVs, 1 for variable and {} for derivatives. But found {} total" .format(ncv, toskip, data.shape[1])
        data = data[:,0:(ncv+1)]

        #check header
        assert np.shape(data)[0] == reduce(lambda x,y: x * y, grid_points, 1), "Number of lines in grid does not match stated grid point number (calculated from NBINS): read {}, grid points = {} => {}".format(np.shape(data)[0], grid_points, reduce(lambda x,y: x * y, grid_points, 1))


        #build the grid
        self.pot = data[:,ncv]
        #switch from fortran order
        self.pot = np.reshape(self.pot, grid_points, order='F')

    def __str__(self):
        return "{} dimension Grid object from {} to {} with {} grid points. Periodic = {}, Types = {}".format(self.dims, self.min, self.max, self.grid_points, self.periodic, self.types)

    def __eq__(self, other):
        if(self.dims != other.dims):
            return False
        if(self.types != other.types):
            return False
        if(np.all(self.grid_points != other.grid_points)):
            return False
        if(not np.all(np.isclose(self.min,other.min))):
            return False
        if(not np.all(np.isclose(self.max,other.max))):
            return False
        if(not np.all(np.isclose(self.pot, other.pot))):
            return False
        return True

    @property
    def dims(self):
        return self.ncv
    
    @property
    def grid_points(self):
        if(self._grid_points is not None):
            return self._grid_points
        if(self.pot is None):
            return ()
        return np.shape(self.pot)        

    @property
    def nbins(self):
        gp = list(self.grid_points)
        if(gp is None or len(gp) == 0):
            return gp
        for i in range(self.dims):
            if(not self.periodic[i]):
                gp[i] -= 1

        return tuple(gp)

        
    @property
    def ncv(self):
        if(self.pot is None):
            return 0
        return len(np.shape(self.pot))

    @property
    def dx(self):
        if(self.pot is None):
            return 0
        return [float(max - min) / nb for max,min,nb in zip(self.max, self.min, self.nbins)]

    def add_cv(self, name, min, max, bin_number, periodic=False, grid_point_number=None):
        """
        Name can be an integer type or named type. The number of grid
        points will run from min to max, inclusive. If it's periodic,
        it's assumed that you do not want the last grid point to wrap
        to the first grid point so it's excluded.
        add_cv(1,min=0,max=3,grid_point_number=4, periodic=T) would
        yield a grid 0, 1, 2 and with non-periodic conditions it would
        be 0,1,2,3.

        The bin number may be specified instead for prettier
        spacing. If a grid runs from 0 to 100, the bin number is 100
        and the grid_point_number is 101 for non-periodic. For bin
        number 100 if a grid runs from 0 to 100, the grid_point_number
        is 100.
        """
        if(grid_point_number is None):
            if(periodic):
                grid_point_number = bin_number
            else:
                grid_point_number = bin_number + 1

        self.min.append(min)
        self.max.append(max)
        self.meshgrid = None
        self.periodic.append(periodic)
        if(type(name) == type("")):
            self.types.append(Grid.cv_type_map[name])
        else:
            self.types.append(name)
        if(self.pot is None):
            self.pot = np.zeros(grid_point_number)
        else:
            self.pot = np.resize(self.pot, self.grid_points + (int(grid_point_number),))

    def add_margin(self, scale,pretty=True):
        """Add a margin around the potential without affecting bin number. Scale should be > 1""" 
        assert len(scale) == self.ncv
        assert scale > 1
        length_diff = [(y - x) * (s) for x,y,s in zip(self.min, self.max, scale)]
        new_min = [x - l / 2 for x,l in zip(self.min, length_diff)]
        new_max = [x + l / 2 for x,l in zip(self.max, length_diff)]
        if(pretty):
            new_min = [round(x * 10) / 10. for x in new_min]
            new_max = [round(x * 10) / 10. for x in new_max]
        self.set_min(new_min)
        self.set_max(new_max)
        print 'Set min to {} and max to {}'.format(self.min, self.max) 

        
            
    def set_min(self,min,adjust_zero=True):
        """Change the mins. Fills with previous boundaries if extending, otherwise crops"""
        g = self.clone()
        self._clear()
        for t,m,x,b,p in zip(g.types, min, g.max, g.nbins, g.periodic):
            self.add_cv(t,m,x,b,p)
        self.add(g)
        #set values to g minimum
        if(adjust_zero):
            self.pot[np.where(self.pot == 0)] = np.max(g.pot)



    def set_max(self,max,adjust_zero=True):
        """Change the maxs. Fills with previous boundaries if extending, otherwise crops"""
        g = self.clone()
        self._clear()
        for t,m,x,b,p in zip(g.types, g.min, max, g.nbins, g.periodic):
            self.add_cv(t,m,x,b,p)
        self.add(g)
        #set values to g minimum
        if(adjust_zero):
            self.pot[np.where(self.pot == 0)] = np.max(g.pot)


    def stretch(self, scale):
        """Change the mesh grid, stretching it out or shrinking it down        
        """
        assert len(scale) == self.ncv
        length_diff = [(y - x) * (s - 1) for x,y,s in zip(self.min, self.max, scale)]
        print length_diff
        self.min = [x - l / 2 for x,l in zip(self.min, length_diff)]
        self.max = [x + l / 2 for x,l in zip(self.max, length_diff)]

    def rescale(self, scale):
        """Simply rescale the mesh grid
        """
        assert len(scale) == self.ncv
        self.min = [x*s for x,s in zip(self.min, scale)]
        self.max = [x*s for x,s in zip(self.max, scale)]
        
        

    def _wrap(self, x, i):
        return x - (self.max[i] - self.min[i]) * floor((x - self.min[i]) / (self.max[i] - self.min[i]))

    def to_index(self, x, i):
        """
        Returns nearst grid index. If the point exceeds the grid size,
        the boundary is returned.
        """
        if(self.periodic[i]):
            x = self._wrap(x,i)

        return max(0, min(self.grid_points[i] - 1, int(floor( (x - self.min[i]) / self.dx[i]))))

    def index_to_coord(self, index):
        return [self.min[i] + self.dx[i] * j for i,j in zip(range(self.ncv), index)]


    def np_to_index(self, x):
        if(sum(self.periodic) == 0):
            return np.fmax(np.zeros(np.shape(x)), np.fmin(np.array(self.grid_points) - 1, np.floor( (x - np.array(self.min)) / np.array(self.dx))))
        else:
            return tuple([self.to_index(x,i) for i,x in enumerate(x)])


    def add_value(self, x, v):
        """
        Add the value v at point x to the grid
        """
        if(len(x) != self.ncv):
            raise ValueError("Dimension of given x vector does not match grid dimension!")
        index = [0 for xi in x]
        for i, xi in enumerate(x):
            if(not (xi >= self.min[i] and xi <= self.max[i])):
                raise IndexError,"Mesh point is not within grid dimension {}: {}, [{}, {}]".format(i, xi, self.min[i], self.max[i])
            index[i] = self.to_index(xi, i)
        self.pot[tuple(index)] += v

    def set_value(self, x, v):
        """
        Set the value of the grid at point x to v
        """
        if(len(x) != self.ncv):
            raise ValueError("Dimension of given x vector does not match grid dimension!")
        index = [0 for xi in x]
        for i, xi in enumerate(x):
            if(self.periodic[i]):
                xi = self._wrap(xi,i)
            if(not (xi >= self.min[i] and xi <= self.max[i])):
                raise IndexError,"Mesh point is not within grid dimension {}: {}, [{}, {}]".format(i, xi, self.min[i], self.max[i])
            index[i] = self.to_index(xi, i)
        self.pot[tuple(index)] = v

    def get_value(self, x):
        """
        Get the value at point x
        """
        if(len(x) != self.ncv):
            raise ValueError("Dimension of given x vector does not match grid dimension!")
        index = [0 for xi in x]
        for i, xi in enumerate(x):
            if(self.periodic[i]):
                xi = self._wrap(xi,i)
            if(not (xi >= self.min[i] and xi <= self.max[i])):
                raise IndexError,"Mesh point is not within grid dimension {}: {}, [{}, {}]".format(i, xi, self.min[i], self.max[i])
            index[i] = self.to_index(xi, i)
        return self.pot[tuple(index)]


    def add(self, other_grid):
        """Add another grid. In place, does not copy
        """
        if(np.shape(self.pot) == np.shape(other_grid.pot)):
            if(self.min == other_grid.min):
                if(self.max == other_grid.max):
                    self.pot += other_grid.pot
                    return

        def do_add(x):
            try:
                self.pot[tuple(x)] += other_grid.get_value(self.index_to_coord(x))
            except IndexError:
                pass
        self._enumerate_grid(do_add)


    @staticmethod
    def _prepend_emit(array, element):
        array_copy = copy.copy(array)
        array_copy.insert(0, element)
        return array_copy

    def set_bin_number(self, new_shape, mode='constant'):
        """
        Set bin number. Mode is passed to scipy.ndimage.interpolation.zoom, so read on that for info.  
        """

        if(type(new_shape) == int):
            new_shape = (new_shape,)
        else:
            new_shape = tuple(new_shape)
        if(np.array_equal(new_shape, self.nbins)):
            return

        gp = []
        for i in range(self.dims):
            if(self.periodic[i]):
                gp.append(new_shape[i])
            else:
                gp.append(new_shape[i] + 1)

        self.set_grid_point_number(tuple(gp), mode)

    def set_grid_point_number(self, new_shape, mode='constant'):
        """
        Change the number of grid points using a spline
        interpolation. Intended more for zooming on a grid than expanding.
        Mode is passed to scipy.ndimage.interpolation.zoom, so read on that for info.  
        """
        if(type(new_shape) == int):
            new_shape = (new_shape)
        else:
            new_shape = tuple(new_shape)
        if(np.array_equal(new_shape, self.grid_points)):
            return
        from scipy.ndimage.interpolation import zoom
        zoom_factor = np.array(new_shape, dtype='float') / self.grid_points
        self.pot = zoom(self.pot, zoom_factor, prefilter=True, mode=mode)
        self.meshgrid = None

    def _enumerate_grid(self, fxn, dim=None, indices=[], reverse=False):
        """Apply fxn over the grid. end_fxn will be called on only
        edges. Dimension 0 changes slowest and dimension N changes
        fastest. This may be reversed by passing in the reverse flag.

        """
        if(dim is None):
            dim = self.ncv - 1        
        effective_dim = dim
        if(reverse):
            effective_dim = self.ncv - dim - 1
        if(dim > 0):
            for i in range(self.grid_points[effective_dim]):
                self._enumerate_grid(fxn, 
                                     dim - 1, 
                                     Grid._prepend_emit(indices, i),
                                     reverse=reverse)
        else:
            for i in range(self.grid_points[effective_dim]):
                if(reverse):
                    fxn(Grid._prepend_emit(indices, i)[::-1])
                else:
                    fxn(Grid._prepend_emit(indices, i))




    def _assign_grid(self, indices, data):
        self.pot[indices] = data[indices]
                
    def _print_grid(self, indices, output):
        for i,j in enumerate(indices):
            output.write('{: 10.8f} '.format(j * self.dx[i] + self.min[i]))        
        output.write('{: 10.8f}\n'.format(self.pot[tuple(indices)]))

    def plot_2d(self, filename, cmap='jet', resolution=None, axis=(0,1), hold=False, vrange=None):
        """This will make a 2D heatmap. The resolution keyword is the bin
        number on the grid. Set the axis being plotted with the axis
        keyword. Hold can be used if you want to add other elements to
        the plot. vrange is the maximum/minimum in the heat map;
        useful for compaing images side-by-side.

        """
        assert self.dims >= 2
        import matplotlib.pyplot as plt
        old_bins = self.nbins
        if(self.dims > 2):
            #integreate along non-plotting axis
            raise NotImplementedError('Not sure how to plot this. Perhaps you should integrate out a dimension first')
            #data = self.pot[:,:,self.nbins[2] / 2]
        else:
            data = self.pot
        if(resolution is not None):
            self.set_bin_number([resolution if x in axis else self.nbins[x] for x in range(self.dims)])
        
        vmin = np.min(data)
        vmax = np.max(data)
        if(vrange is not None):
            vmin = vrange[0]
            vmax = vrange[1]

        if(not hold):
            plt.figure()
        plt.imshow(np.swapaxes(data, 0, axis[0]), interpolation='nearest', cmap=cmap, extent=[self.min[axis[0]], self.max[axis[0]],self.max[axis[1]],self.min[axis[1]]], vmin=vmin, vmax=vmax)
        if(resolution is not None):
            self.set_bin_number(old_bins)
        plt.colorbar()
        if(not hold):
            plt.savefig(filename)

    def bias_to_pmf(self, target_filename=None, bias_factor=None, boltzmann_factor=1):
        """This is used to estimate the PMF from an experiment directed
        metadynamics run (EDM) or metadynamics run. The
        target_filename is the bias target (or None) from EDM. Please
        read this code before using
        """
        if(bias_factor is not None):
            self.pot *= (bias_factor) / (bias_factor - 1)
        if(target_filename is not None):
            t = Grid()
            t.read_plumed_grid(target_filename)
            t.pot *= boltzmann_factor
            self.add(t)
        self.pot -= np.min(self.pot)
        self.pot *= -1.
        

    def plot_2d_region(self, filename, *region_functions):
        """
        This will plot, with less options than plot_2d, to check if the regions are in the correct spot.
        """
        assert self.dims >= 2
        assert NDITER, "numpy nditer unavailable"
        cmap = 'jet'
        axis=(0,1)
        if(self.meshgrid is None):
            self.meshgrid = np.meshgrid(*[np.arange(min, max, dx) for min,max,dx in zip(self.min, self.max, self.dx)], indexing='ij')

        data = np.zeros(np.shape(self.pot))
        for r in region_functions:
            for x in np.nditer(self.meshgrid):
                indexs = self.np_to_index(x)
                if(r(x)):
                    data[tuple(indexs)] = 1

        import matplotlib.pyplot as plt
        old_bins = self.nbins
        plt.imshow(np.swapaxes(data, 0, axis[0]), interpolation='none', cmap=cmap, extent=[self.min[axis[0]], self.max[axis[0]],self.max[axis[1]],self.min[axis[1]]])
        plt.savefig(filename)


    def normalize(self):
        #make sure we don't have gigantic numbers to start
        self.pot -= np.min(self.pot)
        self.pot += np.log(self.integrate_all())

    def integrate_all(self):
        grids = [np.arange(min, max  if p else max + dx / 2, dx) for min,max,dx,p in zip(self.min, self.max, self.dx, self.periodic)]
        Z = np.exp(-self.pot)
        grids.reverse()
        for g in grids:
            Z = simps(Z, g)
        return Z


    def integrate_region(self, region_function):
        """
        Integrates a region given by the function

        region_function will be passed an array giving the coordinates of 
        a single point (N numbers per N dimensions). Simpson's Rule is 
        used for integration.
        """
        
        assert NDITER, "numpy nditer unavailable"
        #make sure we don't have gigantic numbers to start
        self.pot -= np.max(self.pot)
        Z = np.exp(-self.pot)


        if(self.meshgrid is None):
#            self.meshgrid = np.meshgrid(*[np.arange(min, max, dx) for min,max,dx in zip(self.min, self.max, self.dx)], indexing='ij')
            self.meshgrid = np.meshgrid(*[np.arange(min, max  if p else max + dx / 2, dx) for min,max,dx,p in zip(self.min, self.max, self.dx, self.periodic)])
        for x in np.nditer(self.meshgrid):
            indexs = self.np_to_index(x)
            if(not region_function(x)):
                Z[tuple(indexs)] = 0

        grids = [np.arange(min, max  if p else max + dx / 2, dx) for min,max,dx,p in zip(self.min, self.max, self.dx, self.periodic)]
        grids.reverse()
        for g in grids:
            Z = simps(Z,g)
        
        return Z

        

    def write_plumed1_grid(self, output):

        #helper function
        def print_header_array(name, array, output):
            output.write('#! {} '.format(name))
            for a in array:
                output.write('{} '.format(a))
            output.write('\n')


                        
        if(type(output) == type("")):
            output = open(output, 'w')
        output.write('#! FORCE 0\n')
        output.write('#! NVAR {}\n'.format(self.ncv))
        print_header_array('TYPE', self.types, output)

        print_header_array('BIN', self.nbins, output)
        print_header_array('MIN', self.min, output)
        print_header_array('MAX', self.max, output)
        print_header_array('PBC', [1 if x else 0 for x in self.periodic], output)
        self._enumerate_grid(lambda x: self._print_grid(x, output))


    def write_plumed2_grid(self, output, names=None):
        """
        Pass in CV names + label for pot if you'd like, otherwise generic labels/names/fields are generated
        """
        if(type(output) == type("")):
            output = open(output, 'w')
        if(names is not None):
            if(len(names) == 1):
                names = [names for x in xrange(self.dims)]
            elif(len(names) != self.dims + 1):
                raise ValueError('Length of given names does not match dimensions + 1 label for variable')
        else:
            try:
                names = self.names
            except AttributeError:
                names = ['cv_{}'.format(i) for i in xrange(self.dims)]
                names.append('@label')
                
        output.write('#! FIELDS ')
        for n in names:
            output.write('{} '.format(n))
        output.write('\n')

        for i,n in enumerate(names[:-1]):
            if(np.isclose(abs(self.min[i]), pi)):
                output.write('#! SET min_{} {}pi\n'.format(n, '' if self.min[i] > 0 else '-', 'pi'))
            else:
                output.write('#! SET min_{} {}\n'.format(n, self.min[i]))
            if(np.isclose(abs(self.max[i]), pi)):
                output.write('#! SET max_{} {}pi\n'.format(n, '' if self.max[i] > 0 else '-', 'pi'))
            else:
                output.write('#! SET max_{} {}\n'.format(n, self.max[i]))
            output.write('#! SET nbins_{} {}\n'.format(n, self.nbins[i]))
            output.write('#! SET periodic_{} {}\n'.format(n, 'true' if self.periodic[i] else 'false'))
        
        self._enumerate_grid(lambda x: self._print_grid(x, output))


    def add_png_to_grid(self, filename, invert=False):
        if(self.ncv != 2):
            raise ValueError("This method only makes sense on 2D grids. Grid is currently {} dimension".format(self.ncv))
        from pylab import imread, imshow, gray, mean
        a = imread(filename) # read to RGB file
        if(len(np.shape(a)) == 2):
            gray_scale = a
        elif(np.shape(a)[2] == 4):
            gray_scale = mean(a[:,:,0:2],2) * a[:,:,3] # convert to grayscale by multiplication with alpha channel
        else:
            gray_scale = mean(a,2)# convert to gray scale with meana

        if(invert):
            gray_scale = 1 - gray_scale
            
        gray_scale = gray_scale.astype(np.float64)
        gray_scale = np.rot90(gray_scale)
        #too_small = exp(-10)
        #gray_scale[np.where(gray_scale < too_small)] = too_small        
        #self.pot += np.log(gray_scale)
        old_bins = self.nbins
        self.set_bin_number(np.shape(gray_scale))
        self.pot += gray_scale
        self.normalize()
        self.set_bin_number(old_bins)

    def gaussian_blur(self, radius):
        """Blur the image. Radius can be tuple/array or single value which is
        replicated. Should have dimensions; radius is not in bin
        number.

        """
        from scipy.ndimage.filters import gaussian_filter
        if(type(radius) == type(1.0) or type(radius) == type(1)):
            radius = [radius for x in range(self.dims)]
        assert len(radius) == self.dims
        self.pot = gaussian_filter(self.pot, [r / dx for r,dx in zip(radius, self.dx)])

