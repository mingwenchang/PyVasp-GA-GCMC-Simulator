#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        vasp_io2.py -  A central module for GA/GCMC scripts and  
                       vasp-calculation post analysis

                        
DESCRIPTION
        Extract information from POSCAR/CONTCAR and OUTCAR, including 
        ionic positions, energies, forces, vibration frequencies etc.
        
        The read ionic positions from OUTCAR or POSCAR will be stored
        in an Atoms object. Geometric information or operations can 
        easily obtained or proceeded by the object methods.
        
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""
import os, shutil
import numpy as np
import modules.assister as ast
import modules.data as data
from collections import OrderedDict
    
class Atoms:
    def __init__(self, atomtypes=None, natoms=None, positions=None,
                 cell=None, constraints=None):
        """
        Parameters:
            
        atomtypes: chemical symbols of atoms. Can be a string, a list of 
        chemical symbols, or a list of Atom objects.
        
        natoms: number of atoms per atomic species (one number for each atomic
        speices). A list of int. the length of natoms should be
        equal to the length of atomtypes
        
        positions: list of xyz-positions or anything that can be converted to 
        an ndarray of shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2),...].
        
        constraints: Anything that can be converted to an ndarray of shape 
        (n, 3). For vasp, it will do: [('T','T','T'), ('T','T','T'),...].
        
        cell: a 3x3 matrix  
 
        """
        
        self._names = ['atomtypes', 'natoms', 'positions', 'cell', 'constraints']
        
        if isinstance (atomtypes, (list, tuple)) and len(atomtypes) > 0 \
           and isinstance(atomtypes[0], Atoms) :
            """[Atoms1, Atoms2, ...]"""
            atomsobjs = list(atomtypes)
        elif isinstance (atomtypes, Atoms):
             """An Atoms obj"""
             atomsobjs = [atomtypes]
        else:
            """str1 or [str1, str2,..] """
            atomsobjs = None

        if atomsobjs is not None:
            param = atomsobjs[0].get_attribute(self._names)
            atoms = self.__class__(*param)
            others = atomsobjs[1:]
            for other in others:
                atoms.append(other)
            values = atoms.get_attribute(self._names) 
        else:
            if atomtypes is None:
                atomtypes = 'X'
                natoms = 1
                positions = np.array([[0.00, 0.00, 0.00]])
            values = [atomtypes, natoms, positions, cell, constraints]
        
        for i, name in enumerate(self._names):
            self.set_attribute(name, values[i])
            
    def __repr__(self):
        #chemical formula
        cf = self.get_chemical_formula()
        s = "Atoms('%s')" % (cf) 
        return s
    
    def __add__(self, other):        
        objs = [self.copy(), other]
        atomsobj = self.__class__(objs)
        return atomsobj  
    
    def __sub__(self, other):
        #A - B
        othd = other.datomtypes
        rest = OrderedDict()
        for elem, num in zip(self.atomtypes, self.natoms):
            if elem in other.atomtypes:
                n = num - othd[elem]
                if n > 0: 
                    rest[elem] = n
                    #other.remove(elem)
            else:
                rest[elem] =num
                    
        n = np.sum(list(rest.values()))     
        pos = np.empty((n,3)); cons = np.empty((n,3), dtype='object')
        
        start = 0 
        dindexes = self.dindexes           
        for elem in rest:
            n = rest[elem]
            indexes = dindexes[elem][-n:]
            ids=indexes[0]
            ide=indexes[-1]
            pos[start:start+n] = self.positions[ids:ide+1]
            cons[start:start+n] = self.constraints[ids:ide+1]
            start += n
            
        atomtypes = list(rest.keys()) 
        natoms = list(rest.values()) 
        cell = self.cell
        return self.__class__(atomtypes, natoms, pos, cell, cons)
        
    def __len__(self):
        return len(self.positions)
                     
           
    def get_attribute(self, name=None):
        """Get an attribute according to the name """   
        if name is None:
            names = self._names
            return [self.get_attribute(name) for name in names]
        
        if isinstance(name, (list, tuple)):
            names = name
            return [self.get_attribute(name) for name in names]
        
        if name == 'atomtypes':
            return self.atomtypes
        
        if name == 'natoms':
            return self.natoms
        
        if name == 'positions':
            return self.positions
        
        if name == 'constraints':
            if self.get_sdyn():
                return self.constraints
            else:
                return None
        if name == 'cell':
            if self.get_pbc():
                return self.cell
            else:
                return None
    
    def set_attribute(self, name, value):
        """Get an attribute according to the name"""
        if name == 'atomtypes':
            self.atomtypes = value    
        elif name == 'natoms':
            self.natoms = value
        elif name == 'positions':
            self.positions = value
        elif name == 'constraints':
            self.constraints = value  
        elif name == 'cell':
            self.cell = value
     
    def set_atomic_types(self, atomtypes):
        if isinstance (atomtypes, str):
            self._atomtypes = [atomtypes]
        elif isinstance (atomtypes, (tuple, list)):
            self._atomtypes = list(atomtypes)
        else:
            raise ValueError
   
    def get_atomic_types(self):
        return self._atomtypes
    
    atomtypes = property(get_atomic_types, set_atomic_types)

    def set_number_of_atoms(self, natoms):
        if isinstance (natoms, int):
            self._natoms = [natoms]
        elif isinstance (natoms, (tuple, list)):
            self._natoms = list(natoms)
        else:
            raise ValueError
            
    def get_number_of_atoms(self):
        return self._natoms 
    
    natoms = property(get_number_of_atoms, set_number_of_atoms)
    
    def set_atomic_positions(self, positions):
        positions = np.array(positions)
        self._positions = positions.reshape(self.ntotal, 3)
            
    def get_atomic_positions(self):
        return self._positions 
    
    positions = property(get_atomic_positions, set_atomic_positions)
    
    def get_atomic_positions_by_atomtypes(self, atomtypes):
        if isinstance(atomtypes, str):
            atomtypes = [atomtypes]
        
        #remove dublicate
        atomtypes = list(OrderedDict.fromkeys(atomtypes))

        datomtypes = self.datomtypes
        n = sum( [datomtypes[atom] for atom in atomtypes] )
        positions = np.empty((n,3))
        
        start = 0
        for atom in atomtypes:
            num = datomtypes[atom]
            indexes = self.dindexes[atom] 
            positions[start:start+num] = self.positions[indexes]
            start += num
        return positions
        
    def set_atomic_constraints(self, constraints=None):
        if constraints is not None:
            self._sdyn = True
            if isinstance(constraints, (list, tuple, np.ndarray)):
                constraints = np.array(constraints)
            else:
                constraints = np.full((self.ntotal, 3), constraints)
        else:
            self._sdyn = False
            constraints = np.full((self.ntotal, 3), None)
        self._constraints = constraints
                
    def get_atomic_constraints(self):
        return self._constraints
    
    def del_atomic_constraints(self):
        self.set_atomic_constraints(constraints=None)
        
    constraints = property(get_atomic_constraints,
                           set_atomic_constraints,
                           del_atomic_constraints)
    
    def set_cell(self, cell):
        if cell is not None and isinstance(cell, (list, tuple, np.ndarray)) :
            self._pbc = True
            self._cell = np.array(cell)
        else:
            self._pbc = False
            self._cell = np.full((3, 3), None) 
        
    def get_cell(self):
        return self._cell
    
    def del_cell(self):
        self.set_cell(self, None)
               
    cell = property(get_cell, set_cell, del_cell)
    
    #Get cell properties
    def get_cell_volume(self):
        return ast.tripleproduct(self._cell[0], self._cell[1], self._cell[2])
 
    
    def get_cell_lengths(self):
        a_norm  = ast.vectornorm(self._cell[0])
        b_norm  = ast.vectornorm(self._cell[1])
        c_norm  = ast.vectornorm(self._cell[2])
        return a_norm, b_norm, c_norm 
        
    def get_cell_angles(self):
        alpha = ast.angle(self._cell[1], self._cell[2])
        beta  = ast.angle(self._cell[0], self._cell[2])
        gamma = ast.angle(self._cell[0], self._cell[1])
        return alpha, beta, gamma
        
    #dictionalize   
    def get_dict_atomtypes(self):
        keys = self._atomtypes
        values = self._natoms
        return OrderedDict(zip(keys, values))
    datomtypes = property(get_dict_atomtypes)
    
    def get_dict_positions(self):
        refitems = self.datomtypes
        return ast.match(self._positions, refitems)
    dpositions = property(get_dict_positions)
    
    def get_dict_atomic_indexes(self):
        refitems = self.datomtypes
        indexes = list(range(sum(self.natoms)))
        return ast.match(indexes, refitems)
    dindexes = property(get_dict_atomic_indexes)
   
    def get_dict_constraints(self):
        refitems = self.datomtypes
        return ast.match(self._constraints, refitems)
    dconstraints = property(get_dict_constraints)
    
    #Get properties
    def get_pbc(self):
        return self._pbc
    
    def get_sdyn(self):
        return self._sdyn
    
    def get_stru_center(self):
        return np.array(self.positions).sum(0)/len(self.positions)
    
    def get_cell_center(self):
        if self.get_pbc():        
            cc = frac_to_cart(self.cell[0], 
                              self.cell[1], 
                              self.cell[2], 
                              np.array([[0.50, 0.50, 0.50]])) 
        else:
            cc = np.array([[0.00, 0.00, 0.00]])
        return cc 
    
    def get_total_atoms(self):
        return sum(self._natoms)
    ntotal = property(get_total_atoms)
    
    def get_chemical_formula(self):
        #chemical formula
        cf = '' 
        for atom, number in zip(self.atomtypes, self.natoms):
            cf +=atom
            if number > 1:
                cf +=str(number)
        return cf
    
    def get_atomic_masses(self):
        masses = []
        for atom, number in zip(self.atomtypes, self.natoms):
            masses += number * [data.atomic_masses[atom]]
        return masses 
    
    def get_molecular_mass(self):
        m = sum(self.get_atomic_masses())
        return m #unit in amu
    
    def get_center_of_mass(self):
        masses = np.array(self.get_atomic_masses())
        positions = self.get_atomic_positions()
        com = np.dot(masses, positions) / sum(masses)
        return com #unit in amu
    
    def get_center_of_geometry(self):
        return np.array(self.positions).sum(0)/len(self.positions)
            
    def get_moments_of_inertia(self):
        """Get the moments of inertia along the principal axes.

        The three principal moments of inertia are computed from the
        eigenvalues of the symmetric inertial tensor. Periodic boundary
        conditions are ignored. Units of the moments of inertia are
        amu*angstrom**2.
        
        Following codes are from ASE module:
            
        """
        
        com = self.get_center_of_mass()
        positions = self.get_atomic_positions()
        positions -= com  # translate center of mass to the center of mass
        masses = np.array(self.get_atomic_masses())

        # Initialize elements of the inertial tensor
        I11 = I22 = I33 = I12 = I13 = I23 = 0.0
        for i in range(len(self)):
            x, y, z = positions[i]
            m = masses[i]

            I11 += m * (y ** 2 + z ** 2)
            I22 += m * (x ** 2 + z ** 2)
            I33 += m * (x ** 2 + y ** 2)
            I12 += -m * x * y
            I13 += -m * x * z
            I23 += -m * y * z

        I = np.array([[I11, I12, I13],
                      [I12, I22, I23],
                      [I13, I23, I33]])

        evals, evecs = np.linalg.eigh(I)
        return evals    
    
    def get_distance_matrix(self, replacement=True):
        return ast.get_distance_matrix(self._positions, replacement=replacement)  
    dmatrix = property(get_distance_matrix)
    
    """
    def get_coordination_matrix(self, dev=0.25, strict=True):
        dindexes = self.dindexes
        dmatrix=self.get_distance_matrix(replacement=False)  
        cmatrix = np.empty(dmatrix.shape, dtype=bool)
        for atom1 in self.atomtypes:
            for atom2 in self.atomtypes:
                dxy =  data.atomic_covalent_radii[atom1] + data.atomic_covalent_radii[atom2] 
                st1 = dindexes[atom1][0]; end1 = dindexes[atom1][-1]
                st2 = dindexes[atom2][0]; end2 = dindexes[atom2][-1]
                dsub = dmatrix[st1:end1+1, st2:end2+1]
                
                if strict:
                    csub = ( (1-dev) * dxy < dsub) & (dsub <= (1+dev) * dxy ) 
                else:
                    csub = dsub <= (1+dev) * dxy
                
                
                cmatrix[st1:end1+1, st2:end2+1] = csub
        return cmatrix
    cmatrix = property(get_coordination_matrix)
     """   
    def get_dmatrix_of_two_atomtypes(self, atom1, atom2):
        dindexes = self.dindexes
        start_idx1 = dindexes[atom1][0]; end_idx1 = dindexes[atom1][-1]
        start_idx2 = dindexes[atom2][0]; end_idx2 = dindexes[atom2][-1]
        X1 = self.positions[start_idx1:end_idx1+1]
        X2 = self.positions[start_idx2:end_idx2+1]
        dmatrix = ast.get_distance_matrix(X1, X2)  
        return dmatrix
    
    def get_cmatrix_of_two_atomtypes(self, atom1, atom2, cutoff=None):
        dxy = data.atomic_covalent_radii[atom1] + data.atomic_covalent_radii[atom2] 
        dmatrix = self.get_dmatrix_of_two_atomtypes(atom1, atom2)
        
        if cutoff is None:
            cutoff = 1.25 * dxy
            
        if atom1 == atom2:
            for i in range(len(dmatrix)):
                dmatrix[i,i] =cutoff  
 
        cmatrix = dmatrix <  cutoff
        return cmatrix  
    
    def calculate_nbonds_between_two_atomtypes(self, atom1, atom2, cutoff=None):
        cmatrix = self.get_cmatrix_of_two_atomtypes(atom1, atom2, cutoff)
        return np.count_nonzero(cmatrix)

        
    def is_sexy(self, nbonds=1, dev=0.25):
        dindexes = self.dindexes
        dmatrix=self.get_distance_matrix(replacement=False)  
        #cmatrix = np.empty(dmatrix.shape, dtype=bool)
        cmatrix1 = np.empty(dmatrix.shape, dtype=bool) 
        cmatrix2 = np.empty(dmatrix.shape, dtype=bool)  
        for atom1 in self.atomtypes:
            for atom2 in self.atomtypes:
                dxy =  data.atomic_covalent_radii[atom1] + data.atomic_covalent_radii[atom2] 
                st1 = dindexes[atom1][0]; end1 = dindexes[atom1][-1]
                st2 = dindexes[atom2][0]; end2 = dindexes[atom2][-1]
                dsub = dmatrix[st1:end1+1, st2:end2+1]
                
                #To record bonds <= (1+dev) * dxy (i.e. connections) 
                csub =   dsub <= (1+dev) * dxy  
                cmatrix1[st1:end1+1, st2:end2+1] = csub  
                    
                #To record bonds >= (1-dev) * dxy (i.e. not too short)
                csub =    (1-dev) * dxy < dsub 
                cmatrix2[st1:end1+1, st2:end2+1] = csub
                      
                #csub =   ( (1-dev) * dxy < dsub ) & (dsub <= (1+dev) * dxy ) 
                #cmatrix[st1:end1+1, st2:end2+1] = csub
        
        #Check every atom has n bonds at least         
        bonds1 = np.count_nonzero(cmatrix1 , axis=1) - 1 >= nbonds
        #Check every atom has no bonds too short    
        bonds2 = np.count_nonzero(cmatrix2 , axis=1) + 1 >= len(cmatrix2)
        if np.all(bonds1) and np.all(bonds2):
            return True
        else:
            return False

    def get_fractional(self, pos=None):
        if pos is None:
            pos = self.positions
        frac = cart_to_frac(self.cell[0], self.cell[1], self.cell[2], pos)
        return frac
    
    def reset_atomtype_order(self, atomtypes):
        dindexes = self.dindexes
        datomtypes = self.datomtypes
        positions = np.empty(self.positions.shape, dtype=self.positions.dtype)
        constraints = np.empty(self.constraints.shape, dtype=self.constraints.dtype)
        natoms = [ ]
        
        start = 0
        for atom in atomtypes:
            num = datomtypes[atom]
            i, j = dindexes[atom][0],dindexes[atom][-1]
            positions[start: start+num] = self.positions[i:j+1]
            constraints[start: start+num] = self.constraints[i:j+1]
            natoms.append(num)
            start += num
         
        self.atomtypes = atomtypes
        self.natoms = natoms
        self.positions = positions
        self.constraints = constraints


    def truncate(self, atomtypes, natoms=None, mode='tail'):    
        if isinstance(atomtypes, str):
            atomtypes = [atomtypes]
            
        if isinstance(natoms, int):
            natoms = [natoms]
        elif natoms is None:
            datomtypes = self.datomtypes
            natoms =  [datomtypes[atom] for atom in  atomtypes]
            pass

        n = sum(natoms)
        positions = np.empty((n,3))
        constraints = np.empty((n,3), dtype=object)
        
        start = 0
        for atom, num in zip(atomtypes, natoms):
            indexes = self.dindexes[atom] 
            if mode[0].lower() == 't':
                indexes = indexes[-num:]
            elif mode[0].lower() == 'h':
                indexes = indexes[:num]
            
            positions[start:start+num] = self.positions[indexes]
            constraints[start:start+num] = self.constraints[indexes]
            start += num
        
        if not self.get_sdyn():
            constraints = None
            
        if  self.get_pbc():
            cell = self.cell
        else:
            cell = None
        
        return self.__class__(atomtypes, natoms, positions, cell, constraints)        
             
        
    def append(self, other):
        """Extend an atoms object by appending other atoms object"""
        
        #dictionalization for self
        dpos1 = self.get_dict_positions() 
        dcon1 = self.get_dict_constraints()
        
        #dictionalization for other
        dpos2 = other.get_dict_positions() 
        dcon2 = other.get_dict_constraints()
        
        #Combination two atoms objects 
        dpos = ast.combine(dpos1, dpos2) 
        dcon = ast.combine(dcon1, dcon2)
        
        #
        datomtypes = ast.pair_key_and_amount(dpos)
        atomtypes = list(datomtypes.keys())
        natoms = list(datomtypes.values())
        positions = ast.merge(dpos)
        
        #Get sdyn 
        sdyn1 = self.get_sdyn()
        sdyn2 = other.get_sdyn()
        if sdyn1 and sdyn2:
            constraints = ast.merge(dcon)
        else:
            constraints = None
        
        #Get PBC
        pbc1 = self.get_pbc()
        pbc2 = other.get_pbc()
        if pbc1 or pbc2:
            cell = self.cell
        else:
            cell = None
        
        #Update information
        self.atomtypes = atomtypes
        self.natoms = natoms
        self.positions = positions 
        self.constraints = constraints
        self.cell = cell
        
    def remove(self, atom):
        indexes = self.dindexes[atom]
        atometypes = []; natoms = []
        for i, elem in enumerate(self.atometypes):
            if elem != atom:
                atometypes.append(elem)
                natoms.append(self.natoms[i])

        self.atomtypes = atometypes
        self.natoms = natoms
        self.positions = np.delete(self.positions, indexes, axis=0)
        self.constraints = np.delete(self.constraints, indexes, axis=0)

    def pop(self, atom, i=-1, mode='relative'):
        if mode == 'relative':
            """Pop out an atom """
            pidx = self.dindexes[atom][i] 
        else:
            pidx = i 
            pass
        
        datomtypes = self.datomtypes
        datomtypes[atom] -= 1
        if datomtypes[atom] == 0:
            del datomtypes[atom]
            
        self.atomtypes = list(datomtypes.keys())
        self.natoms = list(datomtypes.values())           
        self.positions  = np.delete(self.positions, pidx, axis=0)
        self.constraints  = np.delete(self.constraints, pidx, axis=0)


    def grab(self, atom, i=None):
        """ grab a set of 'X' atoms according to the indices""" 
        
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
        
        if i is None:
            number = len(dpos[atom])
            i = list(range(number))
        elif isinstance(i, int):
            i = [i]
        
        positions = np.take(dpos[atom], i, axis=0)
        constraints = np.take(dcon[atom], i, axis=0)

        atomtypes = atom
        natoms = len(positions)
        
        if not self.get_sdyn():
            constraints = None
            
        if  self.get_pbc():
            cell = self.cell
        else:
            cell = None
        
        atomsobj = self.__class__(atomtypes, natoms, positions, cell, constraints)
        return atomsobj
    
    
    def sort(self, point=None, descending=False):
        """sort atoms using the relative distances between atoms 
           and a specific point.
           The defalut point is the center of the current structure 
        """
        if point is None:
            point = self.get_center_of_geometry()
            
        if descending:
            m = -1 
        else:
            m = 1
                      
        start = 0
        for atom, num in zip(self.atomtypes, self.natoms):
            positions = self.positions[start:start+num]
            constraints = self.constraints[start:start+num]
            indexlist = np.argsort (m * np.linalg.norm(positions-point,axis=1)) 
            self.positions[start:start+num] = positions[indexlist]
            self.constraints[start:start+num] = constraints[indexlist]
            start += num
            
    def align(self, axis='z', descending=True):
        if axis == 'x':
            idx = 0
        elif axis == 'y':
            idx == 1
        else:
            idx = 2
        
        if descending:
            m = -1 
        else:
            m = 1
            
        start = 0
        for atom, num in zip(self.atomtypes, self.natoms):  
            positions = self.positions[start:start+num]
            constraints = self.constraints[start:start+num]
            indexlist = np.argsort(m*positions[:,idx])    
            self.positions[start:start+num] = positions[indexlist]
            self.constraints[start:start+num] = constraints[indexlist]
            start += num
        
    #Manipulations
    def move_to_origin(self):
        """Set the center of a structure to (0.00, 0.00, 0.00)."""   
        self.positions = ast.move_to_origin(self.positions)
        
    def move_to_cell_center(self):
        if self._pbc:
            cc = self.get_cell_center()
            self.positions = ast.move_to_the_point(self.positions, cc)
            pass
        else:
            self.move_to_origin()
    
    ##Move center of geometry to a new point
    def move_to_the_point(self, point):
        #self.positions = ast.move_to_the_point(self.positions, point)
        center = self.get_center_of_geometry()
        movect = point - center
        self.translate(movect)
   
    def translate(self, vect):
        self.positions += vect
        
    def rotate(self, angle=None, axis=None):
        if angle is None:
            angle   = np.random.uniform(0,360)
        if axis is None:
            axis = np.random.choice(['x', 'y', 'z'])
            
        center = self.get_center_of_geometry() 
        self.move_to_origin()
        self.positions = ast.rotate_structure(self.positions, angle, axis)     
        self.move_to_the_point(center)
        
    def euler_rotate(self, phi=None, theta=None, psi=None):
        if phi is None:
            phi   = np.random.uniform(0,360)
        if theta is None:
            theta   = np.random.uniform(0,180)
        if psi is None:
            psi   = np.random.uniform(0,360)
    
        center = self.get_center_of_geometry() 
        self.move_to_origin()
        self.positions = ast. euler_rotate(self.positions, phi, theta, psi)     
        self.move_to_the_point(center)

    def rattle(self, ratio=1.00, delta=0.50, seed=None):
        """Randomly displace atoms.
        
        The displacement matrix is generated from a Gaussian distribution.
        
        delta: Standard deviation (spread or “width”) of the distribution.

        """
        
        rs = np.random.RandomState(seed)
        rdm = rs.normal(scale=delta, size=(self.ntotal, 3))
    
        #Selected atoms randomly 
        nmoves = int(self.ntotal * ratio)
        if nmoves < 1:
            nmoves = 1
        select = np.random.choice(self.ntotal, nmoves, replace=False) #ast.selector(nmoves, ntotal)  
        

        for i in select:
            self.positions[i] += rdm[i]
        
    def sprain(self, ratio=0.50):
        """Randomly selected atoms and rotate them."""
        center = self.get_center_of_geometry() 
        nmoves = int(self.ntotal * ratio)
        
        if nmoves < 1:
            nmoves = 1
        
        self.move_to_origin()
        indices = np.random.choice(self.ntotal, nmoves, replace=False)  #ast.selector(nmoves, ntotal)  
        selected = self.positions[indices]
        
        phi   = np.random.uniform(0,360)
        theta   = np.random.uniform(0,180)
        psi   = np.random.uniform(0,360)
        positions = ast.euler_rotate(selected, phi, theta, psi)  
        

        for i, j in enumerate(indices):
            self.positions[j] = positions[i]
            
        self.move_to_the_point(center)
            
    def twist(self, angle=None, axis=None):
        """split structure into two groups then rotate them randomly"""
        if angle is None:
            theta = np.random.uniform(0,180)
            
        if axis is None:
            axis  = np.random.choice(['x', 'y', 'z'])
        
        h1, h2 = self.split()
        h1.rotate(theta, axis)
        h2.rotate(theta, axis)
        #h1.euler_rotate(); h2.euler_rotate()
        atoms = self.__class__([h1, h2])
        atoms.reset_atomtype_order(self.atomtypes)
        values = atoms.get_attribute()
        for i, name in enumerate(self._names):
            self.set_attribute(name, values[i])
  
    def translocate(self, delta=None, normvect=None):
        if normvect is None:
            normvect = ast.generate_a_normvect('random') 
        if delta is None:
            delta = np.random.uniform(0.5, 1.0)
            
        pp = np.random.randn(3)  # take a random vector
        pp -= pp.dot(normvect) * normvect  # make it orthogonal to normvect
        pp /= np.linalg.norm(pp)  # normalize it     
        
        #the 2nd one:
        #y = np.cross(normvect, pp) # cross product with k

        h1, h2 = self.split(normvect)
        h1.translate(pp+delta)
        atoms = self.__class__([h1, h2])
        values = atoms.get_attribute()
        for i, name in enumerate(self._names):
            self.set_attribute(name, values[i])
 
    def permutate(self, atom1=None, atom2=None, ratio=0.50):
        if atom1 is None and atom2 is None:
            #Randomly exchange atoms.
            i, j = ast.selector(2, len(self.atomtypes))
            atom1 = self.atomtypes[i]
            atom2 = self.atomtypes[j]
            natoms1 = self.natoms[i]
            natoms2 = self.natoms[j]
        else:
            natoms1 = self.datomtypes[atom1]; 
            natoms2 = self.datomtypes[atom2]; 
        
        if natoms1 < natoms2:
            nexchanges = int(natoms1 * ratio)   
            if nexchanges < 1:
                nexchanges = 1 
        else:
            nexchanges = int(natoms2 * ratio) 
            if nexchanges < 1:
                nexchanges = 1
        
        dindexes   = self.dindexes        
        sel1 = np.random.choice(dindexes[atom1], nexchanges, replace=False)
        sel2 = np.random.choice(dindexes[atom2], nexchanges, replace=False)                
        
        for k, l  in zip(sel1, sel2):
            self.positions[k], self.positions[l] = np.copy(self.positions[l]), np.copy(self.positions[k])
            self.constraints[k], self.constraints[l] = np.copy(self.constraints[l]), np.copy(self.constraints[k])
                

    def split(self, normvect=None):
        if normvect is None:
            normvect = ast.generate_a_normvect('random') 
 
        #Save the current geometric center 
        center = self.get_center_of_geometry()
        
        #move the clu to (0, 0 ,0)
        self.move_to_origin()
        
        #Classify atoms in the clu into two parts according to thier angles 
        dot = np.dot(self.positions, normvect)
        n1 = np.linalg.norm(self.positions, axis=1) 
        n2 = np.linalg.norm(normvect, axis=0) #(3,)
        angles = np.arccos(dot/ n1 * n2 ) * (180/np.pi)
        
        #Using Boolean mask 
        leftclu  = self.positions[angles <= 90 ]
        lindx = np.arange(np.sum(self.natoms))[angles <= 90]
        
        rightclu  = self.positions[angles > 90 ] 
        rindx = np.arange(np.sum(self.natoms))[angles > 90]
        
        leftatomtypes = []; rightatomtypes = []
        leftnatoms = []; rightnatoms = []
        start = 0
        for elem, num in zip(self.atomtypes, self.natoms):
            #left part 
            leftnum = len(set(range(start, start+num)) & set(lindx)) 
            if leftnum > 0:
                leftatomtypes.append(elem) 
                leftnatoms.append(leftnum)
     
            #right part 
            rightnum  = len(set(range(start, start+num)) & set(rindx)) 
            if rightnum > 0:
                rightatomtypes.append(elem) 
                rightnatoms.append(rightnum)        
        
            start +=num
            
        left  = self.__class__(atomtypes=leftatomtypes, natoms=leftnatoms, 
                               positions=leftclu, cell=self.cell, constraints='T')
        
        right = self.__class__(atomtypes =rightatomtypes, natoms=rightnatoms,
                               positions=rightclu, cell=self.cell, constraints='T')
        
        self.move_to_the_point(center)
        left.translate(center)
        right.translate(center)
        
        return left, right
          
    def crossover(self, other, stoich=None, normvect=None):
        cent1 = self.get_center_of_geometry()
        cent2 = other.get_center_of_geometry()
        self.move_to_origin()
        other.move_to_origin()
        
        cutatomsobj1 = self.split(normvect)
        cutatomsobj2 = other.split(normvect)

        if  np.random.random() > np.random.random():
            spl_right = cutatomsobj1[0]; spl_left = cutatomsobj2[1]
            res_right = cutatomsobj1[1]; res_left = cutatomsobj2[0]
        else:
            spl_right = cutatomsobj1[1]; spl_left = cutatomsobj2[0]
            res_right = cutatomsobj1[0]; res_left = cutatomsobj2[1]
            
        self.move_to_the_point(cent1)
        other.move_to_the_point(cent2)
        
        #Spliced 
        new = self.__class__([spl_right, spl_left])
        red = self.__class__([res_right, res_left])  
 
        #check stoichemistry  
        if stoich == 'hold':
            stoich = self.datomtypes
            atomtypes = self.atomtypes
        else:
            atomtypes = list(stoich.keys())
            
        new.sort()
        red.sort()            
        for element in stoich:
            if element in new.atomtypes :
                nextra = new.datomtypes[element] -  stoich[element] 
                if nextra > 0 : #Remove nextra atoms which are farthest away from the cutting plane
                    [new.pop(element) for i in range(nextra)]
                elif nextra  < 0:
                    need = red.truncate(atomtypes=element, natoms=abs(nextra), mode='tail')
                    new.append(need)
            else:
                need = red.truncate(atomtypes=element, natoms=stoich[element], mode='tail')
                new.append(need)
                
        new.reset_atomtype_order(atomtypes)
        return new

    def copy(self):
        """Return a copy"""
        atomsobj = self.__class__(self.atomtypes, self.natoms, self.positions, self.cell, self.constraints)
        return atomsobj
    
    def write(self, filename=None, format='xyz', mode='w'):
        """Write """
        if filename is None:
            tag = self.get_chemical_formula()
        else:
            tag = filename
            
        if format == 'xyz':
            if '.xyz' not in tag:
                tag += '.xyz'
            write_xyz(self, filename=tag, mode=mode)
        elif format == 'vasp':
            if  'POSCAR' not in tag and 'CONTCAR' not in tag:
                tag += '.vasp'
            write_poscar(self, filename=tag, mode=mode)

def read_poscar(filename='POSCAR'):
    
    if os.path.exists(filename):
        f = open(filename, 'r')
    else:
        print (filename, "doesn't exit")
    
    # A comment line    
    comment = f.readline() 
    
    #A scaling factor/the lattice constant
    lc = float(f.readline().split()[0])
    
    #Lattice vectors     
    cell = [ ]
    for i in range(3):
        l = f.readline().split()
        vect = float(l[0]), float(l[1]), float(l[2]) 
        cell.append(vect)
    cell = np.array(cell) * lc
    
    #Get atomic types
    atomtypes = [i for i in f.readline().split()] 
    try:
        int(atomtypes[0])
        atomamounts = [int(i) for i in atomtypes]
        atomtypes = comment.split()
    except ValueError:
        atomamounts = [int(i) for i in f.readline().split()] 
    
    #Check selective dynamics tag 
    sdyn_line = f.readline().strip()
    if sdyn_line[0].upper() == 'S':
        sdyn = True
        format_line = f.readline().strip()
    else:
        sdyn = False
        format_line = sdyn_line
    
    #Check formate
    if format_line[0].upper() == 'C' or format_line[0].upper() == 'K':
        cartesian = True
    else:
        cartesian = False 
        
    #Read coordinates, constraints
    total_atoms = sum(atomamounts)
    if sdyn:
        positions = []
        constraints = []  
        for i in range(total_atoms):
            l = f.readline().split()
            vect = float(l[0]), float(l[1]), float(l[2])
            positions.append(vect)
            const = l[3], l[4], l[5]
            constraints.append(const)
        positions = np.array(positions)
        constraints = np.array(constraints)
    else:
        positions = []
        constraints = None
        for i in range(total_atoms):
            l = f.readline().split()
            vect = float(l[0]), float(l[1]), float(l[2])
            positions.append(vect)
        positions = np.array(positions)
            
    #Convert Fractional to Cartesian
    if not cartesian:
        positions = frac_to_cart(cell[0], cell[1], cell[2], positions)
        
    f.close()
    
    atoms = Atoms(atomtypes, atomamounts, positions, cell, constraints)

    return atoms
            
#Convert a set of position vectors to Cartesian from Fractional 
def frac_to_cart(v1, v2, v3, posvects):
    tmatrix = ast.build_tmatrix(v1,v2,v3,'f2c')
    cartesian = np.dot(posvects, tmatrix.T)
    return  cartesian

#Convert a set of position vectors to Fractional from Cartesian
def cart_to_frac(v1, v2, v3, posvects):
    tmatrix = ast.build_tmatrix(v1,v2,v3,'c2f')
    fractional = np.dot(posvects, tmatrix.T)
    return fractional

#Write xyz file
def write_xyz(obj, filename='POSCAR.xyz', mode='w'):
    with open(filename, mode) as xyz:
        xyz.write("%s\n" %(obj.ntotal))
        xyz.write("xyz\n")
        pos = obj.positions 
        start = 0
        for atom, num in zip(obj.atomtypes, obj.natoms):
            for i in range(start, num+start):
                xyz.write("%s %12.6f %12.6f %12.6f\n"%(atom, pos[i][0], pos[i][1], pos[i][2]))
            start += num

#Write POSCAR file
def write_poscar(obj,filename='POSCAR.vasp', tag='Cartesian', ver='vasp5', mode='w'):

    sdyn = obj.get_sdyn()
    cell = obj.get_cell()
    atomtypes = obj.atomtypes
    natoms = obj.natoms
    pos = obj.positions
    con = obj.constraints
    
    with open(filename, mode) as poscar:
        #write title section
        poscar.write("%s\n" %(' '.join(atomtypes)))
        
        #write 1.00 as the scaling factor 
        poscar.write('1.00\n')
    
        #write unscaling lattice vectors
        for i in range(3):
            poscar.write(" %18.15f   %18.15f   %18.15f\n" %(cell[i][0], cell[i][1], cell[i][2]))
            
        #write atom type 
        if ver == 'vasp5':
            poscar.write("%s\n" %('   '.join(atomtypes)))
            
        #Write the number of atoms        
        poscar.write("%s\n" %('   '.join(map(str, natoms))))
    
        if sdyn:
            poscar.write('Selective dynamics\n')
            
        if  tag.upper()[0] == 'C' or tag.upper()[0] == 'K':
            poscar.write('Cartesian\n')
        elif tag.upper()[0] == 'D':
            poscar.write('Direct\n')
            
        #write coordinates and constrains of atoms
        start = 0    
        for atom, num in zip(atomtypes, natoms):
            for i in range(start, num+start):
                if sdyn:
                    poscar.write(" %18.15f %18.15f %18.15f %3s %3s %3s\n" \
                                 %(pos[i][0], pos[i][1], pos[i][2], \
                                   con[i][0], con[i][1], con[i][2]))
                else:
                    poscar.write(" %18.15f %18.15f %18.15f\n" \
                                 %(pos[i][0], pos[i][1], pos[i][2]))
            start += num
 
    
#Functions for reading OUTCAR   
def get_ionic_types(file='OUTCAR'):
    with open(file) as outcar:
        atomtypes = [ ]
        for line in outcar:
            if line.find('POTCAR:' ) > -1:
                atom = line.split()[2]
                atomtypes.append(atom)
                continue
                
            if line.find('W    W    AA    RRRRR' ) > -1:
                break
            #if line.find(' POSCAR =') > -1:
            #    atomtypes = line.split()[2:]
    return atomtypes       
        
def get_number_of_ions_per_type(file='OUTCAR'):
    with open(file) as outcar:
        for line in outcar:
            if line.find('ions per type' ) > -1:
                nions = [int(i) for i in line.split()[4:]]
    return nions   
    
def get_lattice_vectors(file='OUTCAR'):
    with open(file) as outcar:    
        start = False
        n = 0 
        vectors = [ ]
        for line in outcar:
            if line.find('direct lattice vectors') > -1:
                start = True
                continue
            if start:
                vectors.append([float(i) for i in line.split()[0:3]])
                n += 1 
            if n >=3:
                break
    return np.array(vectors)
 
def get_structures(file='OUTCAR', mode=None):
    iontypes = get_ionic_types(file)
    nions = get_number_of_ions_per_type(file)
    cell = get_lattice_vectors(file)
    with open(file) as outcar:    
        start = False
        n = 0 
        strus = []
        positions = []
        for line in outcar:
            if line.find('position of ions in cartesian coordinates') > -1 or \
               line.find('TOTAL-FORCE (eV/Angst)') > -1:
                start = True
                continue
            if start and line.find('--------------') == -1:
                positions.append([float(i) for i in line.split()[0:3]])
                n += 1
            if n >= sum(nions):
                atomsobj = Atoms(iontypes, nions, positions, cell)
                strus.append(atomsobj)   
                start = False
                n = 0
                positions = []
                
    if mode is None:
        strus = strus[-1]
                     
    return strus

#Get DFT energy from OUTCAR
def get_energy(filename = 'OUTCAR', mode=None):
    if mode is None:
        ezero = 999999
        #efree = 999999
    else:
        ezero =  [ ]
        #efree =  [ ]
        
    if  os.path.exists(filename):
        for line in open(filename, 'r'):
            #energy(sigma->0)
            if line.startswith('  energy  without entropy'):
                if mode is None:
                    ezero = float(line.split()[-1])
                else:
                    ezero.append(float(line.split()[-1]))
            """       
            #free energy
            if line.lower().startswith('  free  energy   toten'):
                if mode is None:
                    efree = float(line.split()[-2])
                else:
                    efree.append(float(line.split()[-2]))
            """ 
    else:
        print (filename, ' was not found')
        ezero = 999999
    return ezero
        
#Get Maximum Force from OUTCAR
def get_force(filename='OUTCAR', mode=None):
    if mode is None:
        force = 999999
    else:
        force =  [ ]

    if os.path.exists(filename):
        for line in open(filename, 'r'):
            #FORCES: max atom
            if line.startswith('  FORCES: max atom, RMS'): 
                if mode is None:
                    force = float(line.split()[-2])
                else:
                    force.append(float(line.split()[-2]))
    else:
        print (filename, ' was not found')
        force = 99999
    return force

#Get Eigenvectors and eigenvalues of the dynamical matrix from OUTCAR    
def extra_vibinfo(filename='OUTCAR'):
    if os.path.exists(filename):
        outcar = open(filename,'r')
        start = 'Eigenvectors and eigenvalues of the dynamical matrix'
        end   = 'Finite differences POTIM' 
        sqrt  = 'Eigenvectors after division by SQRT(mass)'
    else:
         raise IOError('%s does not exist!!' %(filename))
        
    infomatrix = []
    switch = False
    for line in outcar:
        line = line.strip()
        if start in line:
             switch = True
        elif sqrt in line:
            switch = None
        elif end in line:
             switch = False
        
        if switch and line !=  '':
               infomatrix.append(line)
        elif switch == None:
            infomatrix = []
        else:
            continue
    infomatrix = infomatrix[2:]
    return infomatrix

#Get frequencies
def get_freqs(infomatrix, unit='cm-1'):
    freqs = []
    freqinfo = [line for line in infomatrix if 'cm-1' in line]
    for line in freqinfo:
        mode = line.split('=')[0] #i.e. 1 f or 1 f/i
        values = line.split('=')[1]  
        
        if unit.lower()=='mev':
            freq = float(values.split()[6])   
        else:
            freq = float(values.split()[4])  #unit in cm-1
    
        if 'f/i' in mode:
            freq = -1 * freq
            
        freqs.append(freq)
 
    return freqs

#Get Eigenvectors of modes 
def get_dymatrix(infomatrix):
    dymatrix = OrderedDict()
    for line in infomatrix:
        if 'X' in line:
            continue
        elif 'f' in line:
            mode = line.split('=')[0].strip().replace(' ','')
            mode = mode.replace('f/i','i')
            dymatrix[mode] = []
        else:
            vector = [float(pos) for pos in line.split()[3:]]
            dymatrix[mode].append(vector)
    return dymatrix    

#whether is a successful calculation
def is_a_successful_vasp_job(filename="OUTCAR"):
    string = 'General timing and accounting informations for this job:'
    info = ast.grep_a_string(string, filename)
    
    if len(info) > 0:
        return True
    else:
        return False

def is_a_converged_vasp_job(filename="OUTCAR", ediffg=0.05):
    force = get_force(filename)         
    if abs(force) <= abs(ediffg):
        return  True
    else:
        return False
 
    
class VaspResultAnalyzer:
    
    """
    Parameters:
            
    workdir: str
        path to the vasp working folder
        
    ediffg: float
        the condition for a converged vasp calculation 
    
    truncations: dict
         define the truncated atoms 
        { 
          "atomtypes": ["atom1", "atom2", "atom3",...],
          "natoms": [int1, int2, int3, ....]
         
         }

    """
    def __init__(self, workdir=None, ediffg=0.05, truncref=None):
        if workdir is None:
            self.set_workdir(os.getcwd())
        else:
            self.set_workdir(workdir)

        self.ediffg = ediffg
        self.truncref = truncref
        
    def __repr__(self):
        name = os.path.basename(self.workdir)
        return "VaspResultAnalyzer(%s)" %(name)   
        
    def get_workdir(self):
        return self._workdir
    
    def set_workdir(self, workdir):
        self._workdir = workdir
        self.outcar = os.path.join(self._workdir, "OUTCAR")   
        self.contcar = os.path.join(self._workdir, "CONTCAR")     
        
        if os.path.exists(self.contcar):
            self._atoms = read_poscar(self.contcar)
            #self._atoms.write(self.poscar, format='vasp')
            #shutil.copy(self.contcar, self.poscar)
        else:
            #print ("No CONTCAR was found...")
            self._atoms  =  Atoms()
        
    workdir = property(get_workdir,  set_workdir)
    
    def set_restart(self):
        self.poscar = os.path.join(self._workdir , "POSCAR")  
        if os.path.exists(self.contcar):
            print ("Copy CONTCAR as POSCAR...")
            #atoms = read_poscar(self.contcar)
            #atoms.write(self.poscar, format='vasp')
            shutil.copy(self.contcar, self.poscar)
        else:
            print ("CONTCAR was not found...")

    def write(self, *args, **kwargs):
        self._atoms.write(*args, **kwargs)
 
    @property    
    def energy(self):
        return get_energy(self.outcar)  
    
    @property
    def force(self):
        return get_force(self.outcar)
    
    @property
    def atoms(self):
        if self.truncref is None:
            return self._atoms  
        elif type(self.truncref) is dict:
            return self._atoms.truncate(atomtypes=self.truncref["atomtypes"], natoms=self.truncref["natoms"])
        else:
            return self._atoms - self.truncref

    #A normal termination job?    
    def is_successful(self):
        return is_a_successful_vasp_job(self.outcar)  
    
    #A converged vasp job? 
    def is_converged(self):
        return is_a_converged_vasp_job(self.outcar, self.ediffg)
   
