#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        examiner.py -  A central module for GA/GCMC scripts

                        
DESCRIPTION
        Assign a fitness to each candidate in a population
        
        Analyze similarity of structures 

DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import numpy as np
import modules.assister as ast
#import gagcmc.vasp_io as vio

class BasicStruExaminer:
    def __init__(self, drel=0.03, dmax=0.07, dene=0.02):
        
        self.drel = drel
        self.dmax  = dmax
        self.dene  = dene
        
    def _is_structurally_identical(self, atomsobj1, atomsobj2):
        if atomsobj1.atomtypes == atomsobj2.atomtypes:
            major  = []; minor = []
            from itertools import combinations_with_replacement as combinations
            for atom1, atom2 in combinations(atomsobj1.atomtypes, 2):
                pos1 = atomsobj1.get_atomic_positions_by_atomtypes(atom1)
                pos2 = atomsobj2.get_atomic_positions_by_atomtypes(atom2)
                boolean = hammer_similarity(pos1, pos2, self.drel, self.dmax)
                
                if atom1 == atom2:
                    major.append(boolean)
                else:
                    minor.append(boolean)
                    
            if np.any(major):
                return True
            elif len(minor) > 0 and np.count_nonzero(minor)/len(minor) > 0.50:
                return True
            else:
                return False
    
        else:
            return False
      
    def is_identical(self, resultParserInstance1, resultParserInstance2):
        ene1 = resultParserInstance1.energy 
        ene2 = resultParserInstance2.energy 
        ediff = abs(ene2 - ene1)
        identical = self._is_structurally_identical(resultParserInstance1.atoms, 
                                                    resultParserInstance2.atoms)

        if identical and ediff < self.dene:
            return True
        else:
            return False        
        
    def is_bad_bridge(self, clu1, clu2, minlength=1.20, maxlength=3.00):
        dmatrix = ast.get_distance_matrix(clu1, clu2)
        dmatrix.sort(axis=1)
        if np.all(dmatrix[:,0] > minlength) & np.all(dmatrix[:,0] < maxlength):
            return False #good 
        else:
            return True #bad  
 
    def get_distances_betwen_two_clus(clu1, clu2):
        return ast.get_distances_betwen_two_clus(clu1, clu2)
             
def hammer_similarity(clu1, clu2, drel=0.03, dmax=0.07):
    """ Hammer's structure examiner:
        
    Two structures are considered equa if their relative accumulated difference
    between the two structures and the maximum difference between two 
    distances are smaller than drel and dmax, respectively. 
    
    cf. https://aip.scitation.org/doi/10.1063/1.4886337        
    """         

    dmatrix1 = ast.get_distance_matrix (clu1)
    dmatrix2 = ast.get_distance_matrix (clu2)
    if dmatrix1.shape == dmatrix2.shape:
        if dmatrix1.shape != (1,1):
            sortdm1  = np.sort(np.triu(dmatrix1, k=1), axis=None)
            sortdm2  = np.sort(np.triu(dmatrix2, k=1), axis=None)  
            maxdist1 = np.nanmax(sortdm1) #Max interatomic distance in cluster1
            maxdist2 = np.nanmax(sortdm2) #Max interatomic distance in cluster2 
            pairdiff = abs(sortdm2 - sortdm1).sum() #Sum(|Di(k) - Dj(k)|) 
            pairsumm = 0.5 * abs(sortdm2 + sortdm1).sum() #0.5*Sum(|Di(k) + Dj(k)|) 
        
            #The telative accumulated difference between the two clusters 
            sigma = pairdiff / pairsumm
            #the maximum difference between two distances in the two clusters
            gamma = abs(maxdist2 -  maxdist1) 
            if sigma < drel and gamma < dmax:
                boolean = True
            else:
                boolean = False   
        else: #only one atom in clus1 and clus2
            boolean = False
    else:
        boolean = False
    return boolean


#----------------Assign a fitness to each candidate in a population------------
#Fitness functions will give a value between 0 - 1 for each candidate in a 
#population. A candidate with a higher fitness value will have a higher 
#opportunity to surive and thrive its genetic codes.   

#Proposed by LLOYD et al.
#cf. https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.20247
def fitfun_tanh(ene, emax, emin):
    rho = (ene - emin)/(emax - emin)
    fitness = 0.5 * (1 - np.tanh(2*rho - 1))
    return fitness


''' 

 #-----------Check a combination is good or bad according to the bonds---------
#This can be used to check a cluster supported on a surface or an alloy 
#whether reasonable             
def is_a_bad_combination(stru1, stru2, cutoff_low, cutoff_up, index=0):
    #Calculate all distances between stru1 and the stru2
    dists = [ ] 
    for atom_i, pos_i in enumerate(stru1):
        for atom_j, pos_j in enumerate(stru2):
            dist = np.linalg.norm(pos_j - pos_i)
            dists.append(dist)
    dists.sort() #sort distances from the shortest to longest 
    
    #dists[0] = the shortest distance between stru1 and stru2
    if dists[0] < cutoff_low or dists[0] > cutoff_up: 
        return True
    #This will make sure that there are n connections between 
    #stru1 and stru2 at least.
    elif  dists[index] > cutoff_up:
        return True
    else:
        return False
    
def is_bad_clu(self, clu, minlength=1.20, maxlength=3.00): 
    if len(clu) == 1 :
        return False
    dmatrix = ast.get_distance_matrix(clu)
    dmatrix.sort()
    if np.all(dmatrix[:,0] > minlength) & np.all(dmatrix[:,0] < maxlength):
        return False #good 
    else:
        return True #bad
    
def is_bad_bridge(self, clu1, clu2, minlength=1.20, maxlength=3.00):
    dmatrix = ast.get_distance_matrix(clu1, clu2)
    dmatrix.sort(axis=1)
    if np.all(dmatrix[:,0] > minlength) & np.all(dmatrix[:,0] < maxlength):
        return False #good 
    else:
        return True #bad    
                  
''' 


''' 
#------------Check a structure is good or bad according to the bonds-----------      
def is_a_bad_structure(stru, lower, upper, i=0): 
    for k, kvect in enumerate(stru):
        bonds = [ ]  #b(i-1), b(i-2), b(i-3), etc.
        for j, jvect in enumerate(stru): 
            if k != j:
                dist =np.linalg.norm(jvect - kvect ) 
                bonds.append(dist)
        bonds.sort()
        
        if  not (lower < bonds[0] < upper):
            """It is not physically reasonable if the shortest distance
            between k and j smaller than the lower limit and larger than
            the upper limit. 
            """
            return True
        
        if not (lower < bonds[i] < upper):
            """Check every atom has n bonds at least. """
            return True
        
    else:
        return False
        

def is_a_bad_structure(stru, cutoff_low, cutoff_up, index=0): 
    #Get the distance matrix of the structure
    dmatrix = ast.get_distance_matrix (stru)
    
    for atom_i, bonds in enumerate(dmatrix):
        bonds = np.delete(bonds, atom_i) #remove 'nan' data in dmatrix
        bonds.sort() #sort bonds from the shortest to longest 
  
        #The bond length < cutoff_low ==> unphysical 
        #The bond length > cutoff_up  ==> unbonded       
        if bonds[0]  < cutoff_low or bonds[0] > cutoff_up: #bonds[0] = the shortest bond of atom_i
            return True     
        #This will make sure that the atom_i has n bonds at least. 
        elif bonds[index] > cutoff_up:
            return True
      
                    
#---------------Check two clusters are identical structures or not-------------
#Two structures are considered equa if their relative accumulated difference
#between between the two structures and the maximum difference between two 
#distances are smaller than drel and dmax, respectively. 
#cf. https://aip.scitation.org/doi/10.1063/1.4886337
def is_identical_structures(cluster1, cluster2, drel = 0.03, dmax = 0.07):

    dmatrix1 = ast.get_distance_matrix (cluster1)
    dmatrix2 = ast.get_distance_matrix (cluster2)
    
    if dmatrix1.shape == dmatrix2.shape:
        if dmatrix1.shape != (1,1):
            sortdm1  = np.sort(np.triu(dmatrix1, k=1), axis=None)
            sortdm2  = np.sort(np.triu(dmatrix2, k=1), axis=None)  
            maxdist1 = np.nanmax(sortdm1) #Max interatomic distance in cluster1
            maxdist2 = np.nanmax(sortdm2) #Max interatomic distance in cluster2 
            pairdiff = abs(sortdm2 - sortdm1).sum() #Sum(|Di(k) - Dj(k)|) 
            pairsumm = 0.5 * abs(sortdm2 + sortdm1).sum() #0.5*Sum(|Di(k) + Dj(k)|) 
        
            #The telative accumulated difference between the two clusters 
            sigma = pairdiff / pairsumm
            #the maximum difference between two distances in the two clusters
            gamma = abs(maxdist2 -  maxdist1) 
            if sigma < drel and gamma < dmax:
                identical = True
            else:
                identical = False   
        else: #only one atom in clus1 and clus2
            identical = True
    else:
        identical = False
        
    return identical
        
#-----------------Check WHOLE structure is identical or not-----------------
def is_identical_structures2(coords1, coords2, drel=0.03, dmax=0.07):
    #The similarity score
    score = 0
    
    #Check each part in coords1 and coords2 is same or not
    keys1 = sorted(list(coords1.keys()))
    keys2 = sorted(list(coords2.keys()))
    if keys1 == keys2:
        for element in keys1:
            if is_identical_structures(coords1[element], coords2[element], drel, dmax):
                score +=  1  
            else:
                score += -99
    else:
        score += -99
        
    #Check entire structures are same or not
    entirety1 = ast.merge(coords1)
    entirety2 = ast.merge(coords2)
    if is_identical_structures(entirety1, entirety2, drel, dmax):
        score +=  1
    else:
        score += -99
        
    if score > 0:  
        identical = True
    else:
        identical = False
       
    return identical
        

#-----------------Check VASP calculation is successful or not-----------------
def is_a_successful_vasp_calculation(filename = 'OUTCAR', ediffg = -0.05):
    force =  vio.get_force(filename)         
    if abs(force) <= abs(ediffg):
        return True


'''





