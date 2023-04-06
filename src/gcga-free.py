#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        ga-supported.py (beta version)

                        
DESCRIPTION

        GA script for searching GM structures of free clusters under a gas enviroment  
        

DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

#DO NOT CHANGE THE FOLLOWING LINES:
#---------------------------------import modules-------------------------------
import os, sys

##import ga-gcmc modules
import numpy as np
import modules.vasp_io2 as vio
import modules.monitor  as mon
import modules.assister as ast
import modules.examiner as exm
import modules.constructor as con
import modules.data as data
   

#----------------------------Define mutation methods-------------------------- 
def dynamicloop(cycles=3):
    def decorator(func):
        def wrapper(*args, **kwagrs): 
            for i in range(np.random.randint(1, cycles)):
                atomsobj = func(*args, **kwagrs)
            return atomsobj
        return wrapper
    return decorator 


class Mutator :
    
    def __init__(self, GAParameterManagerInstance=None):
         self.gaparmgr  = GAParameterManagerInstance
         self.methods = {"rattle":self.rattle, "twist":self.twist, "sprain":self.sprain,
                         "translocate":self.translocate, "permutate":self.permutate,
                         "insert":self.insert, "insert2":self.insert2, "insert3":self.insert3,
                         "delete":self.delete, "delete2":self.delete2}
         
    def __call__(self, atomsobj, operator):
        if len(self.gaparmgr.X1) > 1:
            if operator == "delete":
                operator = "delete2"
            elif operator == "insert":
                operator = np.random.choice(["insert2", "insert3"])
                
        if operator is not None:
            Xfunc = self.methods[operator] 
            mutant = Xfunc(atomsobj.copy())
        else:
            mutant = atomsobj        

        return mutant
      
    def rattle(self, atomsobj):
        """Randomly displace atoms."""
        atomsobj.rattle(ratio=0.50, delta=0.5)
        return atomsobj
        
    def twist(self, atomsobj):
        """split structure into two groups then rotate them randomly"""
        atomsobj.twist(angle=None, axis=None)
        return atomsobj
        
    def translocate(self, atomsobj):
        """split structure into two groups then shift 
        their relative position randomly"""
        normvect = ast.generate_a_normvect('random')
        delta = np.random.uniform(0.30, 0.50)
        atomsobj.translocate(delta=delta, normvect=normvect)
        return atomsobj
        
    def sprain(self, atomsobj):
        """Randomly selected atoms and rotate them."""
        atomsobj.sprain(ratio=0.30)
        return atomsobj 
        
    def permutate(self, atomsobj):
        """ exchange positions of two different atom types """
        atom1, atom2 = np.random.choice(self.gaparmgr.cluster.atomtypes, 2, replace=False)
        atomsobj.permutate(atom1, atom2, ratio=0.30)  
        return atomsobj    
            
    #For single atoms 
    @dynamicloop()
    def insert(self, atomsobj):
        """Randomly insert an atom."""
        x = self.gaparmgr.X1
        s = np.random.choice(self.gaparmgr.cluster.atomtypes)
        dxs = data.atomic_covalent_radii[s] +  data.atomic_covalent_radii[x]   
        cent = atomsobj.get_center_of_geometry()
        pv = cent + con.S_BLDA(mean1=dxs, dev1=0.05, clu=atomsobj.positions - cent) [-1]
        added = vio.Atoms(atomtypes=x, natoms=1, positions=pv, cell=atomsobj.cell, constraints='T' )
        atomsobj.append(added)   
        return atomsobj
    
    @dynamicloop()     
    def delete(self, atomsobj):
        """Randomly delete an atom."""
        x = self.gaparmgr.X1
        if x in atomsobj.atomtypes:
            num = atomsobj.datomtypes[x]
            idx = np.random.randint(0, num)
            atomsobj.pop(x, idx)
        return atomsobj
    
    #For diatomic linear molecule  
    @dynamicloop()
    def insert2(self, atomsobj):
        """Randomly insert a linear molecule."""
        mol = [ x for x in self.gaparmgr.X1]
        x = mol[0];  y = mol[1]
        s = np.random.choice(self.gaparmgr.cluster.atomtypes)
        dxs = data.atomic_covalent_radii[s] +  data.atomic_covalent_radii[x]
        dxy = data.atomic_covalent_radii[x] +  data.atomic_covalent_radii[y]
        #Randomly scale bonds 
        dxs *= np.random.uniform(0.75, 1)
        dxy *= np.random.uniform(0.75, 1)
        
        """Inseration at a a bridge site by S_BLDA method """
        #Truncate the bare clu part 
        pos = atomsobj.get_atomic_positions_by_atomtypes(self.gaparmgr.cluster.atomtypes)
        cc  = ast.get_center_point(pos)
        
        #insert mol to the bare clu part 
        clu =  con.S_BLDA(mean1=dxs, dev1=0.05, clu=pos)
        #the first insertd atom 
        xpv =  clu[-1] 
        #the second insertd  atom  
        nv  = xpv / np.linalg.norm(xpv)  
        ypv = xpv + dxy * nv 
        #append to the origin cluster
        pos = np.vstack((xpv,ypv)) + cc 
        added = vio.Atoms(atomtypes=[x,y], natoms=[1,1], positions=pos, cell=atomsobj.cell, constraints='T' )
        atomsobj.append(added)   
        return atomsobj
    
    #For diatomic linear molecule  
    @dynamicloop()
    def insert3(self, atomsobj):
        """Randomly insert a linear molecule."""
        mol = [ x for x in self.gaparmgr.X1]
        x = mol[0];  y = mol[1]
        s = np.random.choice(gaparameters.cluster.atomtypes)
        dxs = data.atomic_covalent_radii[s] +  data.atomic_covalent_radii[x]
        dxy = data.atomic_covalent_radii[x] +  data.atomic_covalent_radii[y]
        #Randomly scale bonds 
        dxs *= np.random.uniform(0.75, 1)
        dxy *= np.random.uniform(0.75, 1)    
        
        """Inseration at a top site """
        #select a top site atom with a low coordination number 
        indxes  = atomsobj.dindexes[s]
        cmatrix = atomsobj.cmatrix[indxes] #coordination matrix 
        bmatrix = np.count_nonzero(cmatrix, axis=1) #cout number of bonds for each atom
        CN = dict(zip(indxes, bmatrix))
        for i in range(100000):
            idx = np.random.choice(indxes)
            if CN[idx] < 6:  #CN < 6
                break
        atomsobj.move_to_origin() #move to (0,0,0)
        spv = atomsobj.positions[idx] #The top site atom
        nv  = spv / np.linalg.norm(spv) #normalize
        #Give a random rotation in case of all inserted mols in the same plane
        nv  =  ast.euler_rotate(nv, phi = np.random.uniform(0,90), 
                                    theta = np.random.uniform(0,45), 
                                    psi = np.random.uniform(0,90))
        #the first insertd atom:
        xpv = spv + dxs * nv 
        #the second insertd  atom 
        ypv = xpv + dxy * nv
        #append to the origin cluster 
        pos = np.vstack((xpv,ypv))
        added = vio.Atoms(atomtypes=[x,y], natoms=[1,1], positions=pos, cell=atomsobj.cell, constraints='T' )
        atomsobj.append(added)  
        atomsobj.move_to_cell_center()
        return atomsobj
    
    #For diatomic linear molecule  
    @dynamicloop()  
    def delete2(self, atomsobj):
        """Randomly delete a linear molecul."""
        mol = [ x for x in self.gaparmgr.X1]
        x = mol[0];  y = mol[1]
        if x in atomsobj.atomtypes and y in atomsobj.atomtypes:
            dictindxes = atomsobj.dindexes
            
            for i in range(100000):
                #Randomly selcte an X atom, which directly binds with the clu 
                idx = np.random.choice(dictindxes[x])
                #Second: get the connection matrix of the X atom
                where = np.argwhere(atomsobj.cmatrix[idx]).flatten()
                #find all Y atoms have a direct binding with the X atom
                idy = [ i for i in where if dictindxes[y][0] <= i and i <= dictindxes[y][-1]]
                if len(idy) > 0:  #Randomly selcte a Y atom that direct binds with the X atom
                    idy = np.random.choice(idy)
                    break
                else:
                    idy = dictindxes[y][-1]
            #Do deletion         
            atomsobj.pop(x, idx, mode='absolute')
            atomsobj.pop(y, idy-1, mode='absolute')
        return atomsobj
 

#-------------------------------Define decorators------------------------------
#hatching a baby until it has a good shape
def hatching(cycles=100000):
    def decorator(func):
        def wrapper(*args, **kwagrs): 
            for i in range(cycles):
                atomsobj = func(*args, **kwagrs)
                if atomsobj is not None:
                    break
            return atomsobj
        return wrapper
    return decorator 

#check a baby whetherlooks good 
def is_sexy(func):
    def wrapper(*args, **kwagrs): 
        atomsobj = func(*args, **kwagrs)
        if not atomsobj.is_sexy(nbonds=1, dev=0.25):
            atomsobj = None
        return atomsobj
    return wrapper
 
#For GCGA: 
#check a baby whetherlooks good 
def is_single(func):
    def wrapper(self, *args, **kwagrs): 
        atomsobj = func(self, *args, **kwagrs)
        x = self.gaparmgr.X1
        if atomsobj is not None and x in atomsobj.atomtypes:
            #We don't want to see too many x-x bonds in the clu
            nx = atomsobj.datomtypes[x]
            nbonds = atomsobj.calculate_nbonds_between_two_atomtypes(x,x)
            if nbonds > (nx/3):
                atomsobj = None
        return atomsobj
    return wrapper

def reordering(func):
    def wrapper(self, *args, **kwagrs): 
        atomsobj = func(self, *args, **kwagrs)
        x = self.gaparmgr.X1
        atomtypes = self.gaparmgr.cluster.atomtypes
        if x in atomsobj.atomtypes:
            atomtypes = atomtypes+ [x]
        atomsobj.reset_atomtype_order(atomtypes)
        return atomsobj
    return wrapper
 
#--------------------------------make a baby  -------------------------------    
class Generator:
    def __init__(self, GAParameterManagerInstance=None):
        self.gaparmgr  = GAParameterManagerInstance
        self.mutator = Mutator(self.gaparmgr)
        self.operator  = None
        self.operations = ["rattle", "twist", "sprain", "translocate"]
        if len(self.gaparmgr.cluster.atomtypes) > 1:
            self.operations.append('permutate')   
        
    #Randomly select an operator     
    def get_operator(self):
        pins, pdel, ptrans =  self.gaparmgr.idtrate
        p = np.random.dirichlet(np.ones(3),size=1).reshape(3,)
        if pins > p[0]:
            self.operator = 'insert'
        elif pdel > p[1]:
            self.operator = 'delete'
        elif ptrans  > p[2]:
            self.operator = np.random.choice(self.operations) 
        return self.operator          
        
    @hatching(1000000)
    @is_single
    @is_sexy
    def primitive(self):
        baby = vio.read_poscar('POSCAR')
        baby.atomtypes = self.gaparmgr.cluster.atomtypes 
        baby.natoms = self.gaparmgr.cluster.natoms
        baby.positions = self.gaparmgr.cluster.generate()
        baby.constraints = [['T', 'T', 'T']] * sum(baby.natoms)
        baby.move_to_cell_center()   
        return baby
    
    @reordering
    @hatching(1000000)
    @is_single
    @is_sexy    
    def offspring(self, mom, dad):
        mom.euler_rotate()
        dad.euler_rotate()
        normvect = ast.generate_a_normvect('random')
        baby = mom.crossover(dad, stoich=self.gaparmgr.cluster.datomtypes, normvect=normvect)
        baby = self.mutator(baby, self.operator)
        baby.move_to_cell_center()
        return baby
    
#---------------------------------Main part ----------------------------------
if __name__ == '__main__':
     
    #Read parameter for GA 
    gainp = sys.argv[1] #'/Users/mwchang/Desktop/BH_GCMC_work/ga.yml'
    gaparameters = mon.GAParametersManager(gainp) 
    
    #Set structure examiner 
    struexaminer = exm.BasicStruExaminer(drel=gaparameters.drel, 
                                         dmax=gaparameters.dmax,
                                         dene=gaparameters.dene)
    #Set generator:
    generator = Generator(gaparameters)  
   
    #Set vassp result analyzer
    bare = vio.read_poscar('POSCAR')
    vaspanalyzer = vio.VaspResultAnalyzer(ediffg=gaparameters.force, truncref=bare)  
  
    #Set ga parallel controller
    gacontroller = mon.GAParallelController(gaparameters, vaspanalyzer, struexaminer)
   
    #Set ga result analyzer
    gaanalyzer = mon.GCGAResultAnalyzer(gaparameters, vaspanalyzer)

    #------------------------ Start GA simulation ------------------------#
        
    gacontroller.initialize()     
    for cyc in range(1, gaparameters.maxgacycle + 1):
        print('\n#Start GA Cycle: %s' %(cyc))
        nthgen  = gacontroller.nthgen + 1 
        gacontroller.mkgendir("gen_%s" %(str(nthgen).zfill(3)))
        
        if gacontroller.ncandiates < gaparameters.genpopsize:
            print('Generating an initial population...')
            for j in range(1, gaparameters.nparallels + 1):
                gacontroller.prepoptdir("stru_%s" %(str(j).zfill(3)))
                print ('A baby is being generated')
                baby = generator.primitive()
                print('Baby: %s/POSCAR' %(gacontroller.optdir))
                baby.write(os.path.join(gacontroller.optdir, 'baby'), format='vasp')
                baby.write(os.path.join(gacontroller.optdir, 'POSCAR'), format='vasp') 
        else:
            if cyc == 1: #A restart GA job
                gaanalyzer.analyze()
                gaanalyzer.save()  
                
            print('Generating a new population......')
            for j in range(1, gaparameters.nparallels + 1):
                gacontroller.prepoptdir("stru_%s" %(str(j).zfill(3))) 
                   
                while True: #Human reproduction
                    #Select parents
                    #i, j = ast.selector2(2, gacontroller.ncandiates, gaanalyzer.competitiveness) 
                    i, j = ast.selector2(2, gaparameters.genpopsize, gaanalyzer.competitiveness) 
                    mom  = gaanalyzer.candidates[i]
                    dad  = gaanalyzer.candidates[j]
                    
                    print('Mimicking human reproduction by crossover:')
                    print('Mother(%.3f): %s/CONTCAR' %(mom.fitness, mom.workdir))
                    print('Father(%.3f): %s/CONTCAR' %(dad.fitness, dad.workdir))
                    print('A baby is being generated...')
                    
                    mom  = mom.atoms
                    dad  = dad.atoms
                    
                    #Mutation? 
                    if gaparameters.mutationrate >  np.random.random():
                        operator = generator.get_operator()
                        print ('Mutation: %s occurs!!' %(operator))
                    else:
                        operator = 'crossover'
                    
                    #Hatching 
                    baby = generator.offspring(mom=mom, dad=dad)
                    
                    if baby is None:
                        print ('Current parents are infertile')
                        print ('Trying to select new parents')
                    else:
                        break
                    
                print('Baby: %s/POSCAR' %(gacontroller.optdir))
                mom.write(os.path.join(gacontroller.optdir,  'mom'), format='vasp')      
                dad.write(os.path.join(gacontroller.optdir,  'dad'), format='vasp') 
                baby.write(os.path.join(gacontroller.optdir, 'baby'), format='vasp')
                baby.write(os.path.join(gacontroller.optdir, 'POSCAR'), format='vasp') 
                
        gacontroller.submit_parallel_jobs()
        gacontroller.monitor()
        gaanalyzer.analyze()
        gaanalyzer.save()     
         
        print('The current population list:')
        print('{:<10}{:<18}{:<18}{:<21}{:<21}'.format('#Rank',
                                                      'Free Energy',
                                                      'Relative Energy', 
                                                      'Prob.(%s K)' %(gaparameters.temperature),
                                                      'Path to Candidate'))
        
        for i, cand in enumerate(gaanalyzer.candidates):
                print('{:<10}{:< 18.8f}{:< 18.8f}{:< 18.8f}{}'.format(i+1, 
                                                                        cand.free, 
                                                                        cand.free-gaanalyzer.gm.free, 
                                                                        cand.propulation,
                                                                        cand.workdir)) 
                
        print ('Remaining GA cycles: %s\n' %(gaparameters.maxgacycle - cyc)) 

    print('GA simulation has done.')
    print('Global minimum structure: %s/CONTCAR' %(gaanalyzer.gm.workdir))



