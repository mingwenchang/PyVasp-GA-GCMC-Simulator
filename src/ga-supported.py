#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        ga-supported.py 

                        
DESCRIPTION

        GA script for searching GM structures of supported clusters   
        

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
#import modules.constructor as con
#import modules.data as data
    

#----------------------------Define mutation methods--------------------------
class Mutator :
    
    def __init__(self, GAParameterManagerInstance=None):
         self.gaparmgr  = GAParameterManagerInstance
         self.methods = {"rattle":self.rattle, "twist":self.twist, "sprain":self.sprain,
                         "translocate":self.translocate, "permutate":self.permutate}
        
    def __call__(self, atomsobj, operator):
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
        normvect = ast.generate_a_normvect('XY')
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

#---------------------------------- decorators--------------------------------
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

#Landing a cluster to the slab
def landing(wag=True):
    def decorator(func):
        def wrapper(self, *args, **kwagrs):
            atomsobj = func(self, *args, **kwagrs)
            if atomsobj is not None:
                landingat = vio.frac_to_cart(slab.cell[0], slab.cell[1],slab.cell[2], self.gaparmgr .landingat)
                if wag:
                    normv = ast.generate_a_normvect('XY')
                    f = np.random.uniform(0.30, 0.50)
                    landingat += f*normv
                self.gaparmgr .cluster.positions = atomsobj.positions
                self.gaparmgr .cluster.land(deck=slab.positions, point=landingat, altitude=self.gaparmgr .altitude, nanchors=gaparameters.nanchors)
                atomsobj.positions = self.gaparmgr.cluster.positions
                atomsobj = slab + atomsobj  
            return atomsobj
        return wrapper
    return decorator 

#---------------------------------make a baby  -------------------------------  
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
        self.operator = np.random.choice(self.operations)    
        return self.operator          
        

    @landing(wag=False)
    @hatching(1000000)
    @is_sexy
    #For generating initial population
    def primitive(self):
        baby = vio.read_poscar('POSCAR')
        baby.atomtypes = self.gaparmgr.cluster.atomtypes 
        baby.natoms = self.gaparmgr.cluster.natoms
        baby.positions = self.gaparmgr.cluster.generate()
        baby.constraints = [['T', 'T', 'T']] * sum(baby.natoms)
        baby.move_to_cell_center()   
        return baby

    @landing(wag=True)
    @hatching(100000)
    @is_sexy
    #For mimicking human reproduction
    def offspring(self, mom, dad, operator=None):
        mom.euler_rotate()
        dad.euler_rotate()
        normvect = ast.generate_a_normvect('random')
        baby = mom.crossover(dad, stoich=self.gaparmgr.cluster.datomtypes, normvect=normvect)
        baby = self.mutator(baby, operator)
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
    slab = vio.read_poscar('POSCAR')
    vaspanalyzer = vio.VaspResultAnalyzer(ediffg=gaparameters.force, truncref=slab)  
  
    #Set ga parallel controller
    gacontroller = mon.GAParallelController(gaparameters, vaspanalyzer, struexaminer)
   
    #Set ga result analyzer
    gaanalyzer = mon.GAResultAnalyzer(gaparameters, vaspanalyzer)


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
                print ('A baby is being generated by SBLDA')
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
                    i, j = ast.selector2(2, gacontroller.ncandiates, gaanalyzer.competitiveness) 
                    mom  = gaanalyzer.candidates[i]
                    dad  = gaanalyzer.candidates[j]
                    
                    print('Mimicking human reproduction by crossover:')
                    print('Mother(%.3f): %s/CONTCAR' %(mom.fitness, mom.workdir))
                    print('Father(%.3f): %s/CONTCAR' %(dad.fitness, dad.workdir))
                    print('A baby is being generated....')
                    
                    #Mutation? 
                    if gaparameters.mutationrate >  np.random.random():
                        operator = generator.get_operator()
                        print ('Mutation: %s occurs!!' %(operator))
                    else:
                        operator = 'crossover'
                    
                    #Hatching 
                    baby = generator.offspring(mom=mom.atoms, dad=dad.atoms)

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
                                                      'Total Energy',
                                                      'Relative Energy', 
                                                      'Prob.(%s K)' %(gaparameters.temperature),
                                                      'Path to Candidate'))
        
        for i, cand in enumerate(gaanalyzer.candidates):
                print('{:<10}{:< 18.8f}{:< 18.8f}{:< 18.8f}{}'.format(i+1, 
                                                                        cand.energy, 
                                                                        cand.energy-gaanalyzer.gm.energy, 
                                                                        cand.propulation,
                                                                        cand.workdir))

        print ('Remaining GA cycles: %s\n' %(gaparameters.maxgacycle - cyc))     

    print('GA simulation has done.')
    print('Global minimum structure: %s/CONTCAR' %(gaanalyzer.gm.workdir))




