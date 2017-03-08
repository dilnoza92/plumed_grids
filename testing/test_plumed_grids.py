from __future__ import absolute_import, division, print_function

import unittest
import sys
import numpy as np
import numpy.testing as npt 
import textwrap

sys.path.append('..')

PLUMED_GRID=8
PLUMED_GRID_SRC='grids'
PLUMED2_GRID=1
PLUMED2_GRID_SRC='grids2'

from plumed_grids import *

class TestPlumedGrid(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_init(self):
        g = Grid()
        
    def test_add_cv(self):
        g = Grid()
        g.add_cv('Coordination number', 0, 3, 5)
        g.add_cv('Coordination number', 0, 3, 5)
        self.assertEqual(g.nbins[0], 5)
        self.assertEqual(np.shape(g.pot), (6,6))
        self.assertEqual(g.grid_points, (6,6))

        g = Grid()
        g.add_cv('Coordination number', 0, 3, 5, periodic=True)
        g.add_cv('Coordination number', 0, 3, 5, periodic=True)
        self.assertEqual(g.nbins[0], 5)
        self.assertEqual(np.shape(g.pot), (5,5))
        self.assertEqual(g.grid_points, (5,5))

    def test_dx(self):
        g = Grid()
        g.add_cv('Coordination number', 0, 10, 10)
        npt.assert_approx_equal(g.dx[0],1)

        g = Grid()
        g.add_cv('Coordination number', 0, 10, 10, True)
        npt.assert_approx_equal(g.dx[0],1)
        

    def test_clone(self):
        g = Grid()
        g.add_cv('Coordination number', 0, 3, 5)
        g.add_cv('Coordination number', 0, 3, 5)
        h = g.clone()
        self.assertFalse(h is g)
        self.assertEqual(h.__str__(), g.__str__())
        self.assertTrue(np.all(h.pot[:] == g.pot[:]))

    def test_to_index(self):
        g = Grid()
        g.add_cv('Distance', 0, 100, 100)
        self.assertEqual(g.to_index(0,0) , 0)
        self.assertEqual(g.to_index(49.5,0) , 49)
        self.assertEqual(g.to_index(49.6,0) , 49)
        self.assertEqual(g.to_index(99,0) , 99)
        self.assertEqual(g.to_index(100,0) , 100)

        g.add_cv('Distance', 0, 10, 10, True)
        self.assertEqual(g.to_index(-0.1,1) , 9)
        self.assertEqual(g.to_index(-1,1) , 9)
        self.assertEqual(g.to_index(-1.01,1) , 8)
        self.assertEqual(g.to_index(10,1) , 0)
        self.assertEqual(g.to_index(11,1) , 1)

    def test_np_to_index(self):

        #this one was failing previously
        g = Grid()
        g.add_cv('Distance', 0, 2, 2)
        g.add_cv('Distance', 0, 2, 2)
        self.assertTrue(np.all(g.np_to_index([0,1]) == (0,1)))
        

        #test w/o periodicity
        g = Grid()
        g.add_cv('Distance', 0, 100, 100)
        g.add_cv('Distance', 0, 100, 100)
        g.add_cv('Distance', 0, 100, 100)

        self.assertTrue(np.all(g.np_to_index([0,1,100]) == (0,1,100)))

        #compare both methods
        self.assertEqual(g.np_to_index([92,21.3,27.2])[1], g.to_index(21.3, 1))

        #with periodidicity
        g = Grid()
        g.add_cv('Distance', 0, 10, 10)
        g.add_cv('Distance', 0, 10, 10, True)
        g.add_cv('Distance', 0, 10, 10)

        self.assertEqual(g.np_to_index([0,1,2]) , (0,1,2))
        self.assertEqual(g.np_to_index([0,-0.5,2]) , (0,9,2))
        self.assertEqual(g.np_to_index([0,10.5,0])[1], g.to_index(10.5, 1))
        self.assertEqual(g.np_to_index([0,11,0])[1], g.to_index(11, 1))
        self.assertEqual(g.np_to_index([0,-2.1,0])[1], g.to_index(-2.1, 1))


    def test_get_set_value(self):
        g = Grid()
        g.add_cv('Distance', 0, 10, 10)
        g.set_value([5], 5)
        npt.assert_approx_equal(g.get_value([5]),5)

        g = Grid()
        g.add_cv('Distance', 0, 10, 10, True)
        g.set_value([-1.1], 1)
        npt.assert_approx_equal(g.get_value([8.9]), 1)


        g = Grid()
        g.add_cv('Distance', 0, 10, 10, False)
        g.add_cv('Distance', 0, 10, 10, True)
        g.add_cv('Distance', 0, 10, 10, True)
        g.add_cv('Distance', 0, 10, 10, False)
        g.set_value([10, -1, -3, 5], 1)
        npt.assert_approx_equal(g.get_value([10, 9, 7, 5]),1)


    def test_set_bin_number(self):
        g = Grid()
        g.add_cv('Distance', 0, 10, 10)        
        g.set_value([0.5], 1)
        g.set_bin_number(100)
        self.assertEqual(np.shape(g.pot)[0], 101)
        npt.assert_approx_equal(g.pot[0], 1)
        self.assertTrue(np.abs(g.pot[9]) < 0.2)
        
    def test_load_data(self):
        g = Grid()
        g.add_cv('Distance', 0, 2, 2)
        g.add_cv('Distance', 0, 2, 2)
        with open('test', 'w') as f:
            f.write(textwrap.dedent('''
            0 0 1
            0 1 2
            0 2 3
            1 0 4
            1 1 5
            1 2 6
            2 0 7
            2 1 9
            2 2 10'''))            
        g.load_data('test')
        npt.assert_approx_equal(g.pot[0,0],1)
        npt.assert_approx_equal(g.pot[1,0], 4)
        npt.assert_approx_equal(g.pot[1,1], 5)
        npt.assert_approx_equal(g.get_value( [1.5,1.5] ), 5)

    def test_load_data_periodic(self):
        with open('test', 'w') as f:
            f.write(textwrap.dedent('''
            0 0 1
            0 1 2
            0 2 1
            1 0 4
            1 1 5
            1 2 4
            2 0 1
            2 1 2
            2 2 1'''))            

        g = Grid()
        g.add_cv('Distance', 0, 2, 2, periodic=True)
        g.add_cv('Distance', 0, 2, 2, periodic=True)
        g.load_data('test')

        print(g.pot[1,0])
        npt.assert_approx_equal(g.pot[0,0], 1)
        npt.assert_approx_equal(g.pot[1,0], 4)
        npt.assert_approx_equal(g.get_value( [-1,-1] ), 10)
        npt.assert_approx_equal(g.get_value( [2, 2] ), 10)
        npt.assert_approx_equal(g.get_value( [3, 3] ), 5)



    def test_load_data_expand(self):
        g = Grid()
        g.add_cv('Distance', 0, 1, 1)
        g.add_cv('Distance', 0, 1, 1)
        with open('test', 'w') as f:
            f.write(textwrap.dedent('''
            0 0 4
            0 1 0
            0 2 0
            1 0 4
            1 1 0
            1 2 0
            2 0 4
            2 1 0
            2 2 0'''))            
        g.load_data('test')
        self.assertEqual(g.nbins , (1,1))
        npt.assert_approx_equal(abs(g.pot[0,0]), 4)

    def test_add(self):
        g = Grid()
        g.add_cv('Distance', 5, 10, 5)
        g.pot[:] = 1

        h = Grid()
        h.add_cv('Distance', 5, 7, 5)
        h.pot[:] = 1
        
        g.add(h)
        
        npt.assert_approx_equal(abs(g.get_value([5])), 2)
        npt.assert_approx_equal(abs(g.get_value([7])), 2)
        npt.assert_approx_equal(abs(g.get_value([8])), 1)
        npt.assert_approx_equal(abs(g.get_value([10])), 1)
        

    def test_set_min_max(self):
        g = Grid()
        g.add_cv('Distance', 5, 10, 5)
        g.set_min((0,))
        self.assertEqual(g.min[0], 0)
        self.assertEqual(g.nbins[0], 5)
        self.assertEqual(g.dx[0], 2)

    def test_normalize(self):
        g = Grid()
        g.add_cv('Distance', 2.5,7.5,10)
        g.add_cv('Distance', 0,10,10)
        g.add_cv('Distance', -3,5,10)
        g.pot = np.random.rand(*np.shape(g.pot))

        g.normalize()
        grids = [np.arange(min, max + 0.00000001, dx) for min,max,dx in zip(g.min, g.max, g.dx)]        
        Z = np.exp(-g.pot)
        grids.reverse()
        for g in grids:
            Z = simps(Z, g)
        self.assertTrue((Z - 1.0)**2 < 0.0001)



    def test_stretch(self):
        g = Grid()
        g.add_cv('Distance', 2.5,7.5,10)
        g.add_cv('Distance', 0,10,10)

        g.stretch([2,1])
        
        self.assertEqual(g.min[0], 0)
        self.assertEqual(g.max[0], 10)
        self.assertEqual(g.min[1], 0)
        self.assertEqual(g.max[1], 10)
        self.assertEqual(g.nbins[0], 10)
        self.assertTrue(abs(g.dx[0]), abs(g.dx[1]))

    def test_plumed_grid_consistency(self):
        for i in range(1, PLUMED_GRID+1):
            input =  '{}/{}.dat'.format(PLUMED_GRID_SRC, i)
            output = '{}.test'.format(input)
            g = Grid()
            g.read_plumed_grid(input)
            g.write_plumed1_grid(output)
            h = Grid()
            h.read_plumed_grid(output)
            if(not h.__eq__(g)):
                print(h == g)
                print("Error on {}".format(output))
            self.assertEqual(h, g)

    def test_plumed2_grid_consistency(self):
        for i in range(1, PLUMED2_GRID+1):
            input =  '{}/{}.dat'.format(PLUMED2_GRID_SRC, i)
            output = '{}.test'.format(input)
            g = Grid()
            g.read_plumed_grid(input)
            g.write_plumed2_grid(output)
            h = Grid()
            h.read_plumed_grid(output)
            if(not h.__eq__(g)):
                print(h == g)
                print("Error on {}".format(output))
            self.assertEqual(h, g)

            
            
if __name__ == '__main__':
    unittest.main()
