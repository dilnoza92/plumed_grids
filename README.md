Plumed Grids
====

A Python script to aid creating and manipulating the grids which are
used by plumed and plumed2. Just add the location of the folder to
your PYTHONPATH with

    export PYTHONPATH=$PYTHONPATH:/path/to/this/repo


It does some nice things, but doesn't work with derivatives and does
not do spline interpolation.


Quickstart
----

```python
from plumed_grids import *

#create empty grid
g = Grid()

#read in a PLUMED grid. In this example, it's assumed to be 2D
g.read_plumed1_grid('BIAS')

#Add a pretty sunset to the bias
g.add_png_to_grid('sunset.png')

#normalize it so it integrates to 1
g.normalize()
print g.integrate_all()

#plot it to make sure it looks good
g.plot_2d('my_bias.png')

#write it to be ready for plumed2
g.write_plumed2_grid('plumed_2.dat')
```

The Grid Object
---

```python


g = Grid()

#Create grid by adding CVs. The name doesn't matter so much, but was tracked in PLUMED1 (not PLUMED2)
#After name comes min, max, nbins, and periodicity
g.add_cv('Absolute position', -155, 155, 1024, False)
g.add_cv('Absolute position', -155, 155, 1024, False)

#add some values to bias using an image
g.add_png_to_grid('test.png')

#see a summary 
print g

#see individual attributes
print g.nbins
print g.dims
print g.dx
print g.ncv


#Write/Read
g.write_plumed1_grid('test_1.dat')
g2 = Grid()
g2.read_plumed1_grid('test_1.dat')

#uses fuzzy floating point to test equality
#useful for testing (perhaps one day for unit tests of PLUMED/PLUMED2)
print g == g2

```

Integrals
---

It is always assume that what is stored in the grid is -ln U, so that
to integrate we take e^(-U) of the values stored.


Regions
----

```python

#Continuing the example from above:

#Create a function which returns True when in region or False
def ellipse_region(x, center, radii):
    sum = 0
     for xi, ci, ri in zip(x, center, radii):
	 sum += (xi - ci) ** 2 / ri ** 2
	      if(sum > 1):
	          return False
	  return True
				    
#create a few particular ellipses
region1 = lambda x: ellipse_region(x, [0,0], [2.5, 5])
region2 = lambda x: ellipse_region(x, [-2,4], [2.5, 5])

#plot them to make sure they're correct
g.plot_2d_region('region1.png', region1)
g.plot_2d_region('region2.png', region2)

#get integrated difference between them
g.integrate_region(region1) - g.integrate_region(region2)
```

PLUMED 1 vs PLUMED 2
----

Note that the derivatives will not be transferred!

```python
g = Grid()
g.read_plumed1_grid('some_grid.dat')
g.write_plumed2_grid('some_grid2.dat')
```


Manipulate Grids
----

```python
g = Grid()

#read in grid previosuly created to get its min/max/bins 
g.read_plumed1_grid('bias.dat')

#zero out its potential
g.pot[:,:] = 0

#add a picture to it
g.add_png_to_grid('logo.png')
g.normalize()

#Make it so that the maximum difference is 3 kT
diff = np.max(g.pot) - np.min(g.pot)
g.pot *= 3. / diff
g.pot -= np.max(g.pot)
g.plot_2d('logo-bias.png', axis=(0,1))
g.write_plumed1_grid(open('logo.dat', 'w'))
g.pot *= -kt
g.pot -= np.max(g.pot)
g.plot_2d('logo-fes.png',axis=(0,1))

######################

g = Grid()
g.read_plumed2_grid('bias.dat')

#Change the bin number with interpolation
g.set_bin_number( (1024, 1024) )

#Stretch by 20% 
g.stretch( (1.2, 1.2) )

#Smooth the bias by adding a blur with a radius of 3 in x and 2 in y
g.gaussian_blur( (3,2) )

#Extend the edge value outwards 20% so that there is a margin. Make it a whole number (pretty = True)
g.add_margin( (1.2, 1.2), pretty=True)

```


Requirements
---

Numpy, Scipy, Matplotlib (if plotting is used)



Disclaimer
----

Grid orientation can be a funny thing. Please make sure by examining
the min/max and number of bins that your grid is correctly oriented in
PLUMED/PLUMED2 and when read into the python code. Plotting a lot
helps too.

Cite
----

This was created for helping in the experiment directed targeted
metadynamics work. Please cite:

**Designing Free Energy Surfaces that Match Experimental Data with Metadynamics**. AD White, JF Dama, GA Voth. *J. Chem. Theory Comput.* **2015**, *11 (6)*, pp 2451-2460
