#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 20 15:47:38 2018

@author: M.W. Chang

Assister Module contains a set of simple functions that allow users/programmer 
to develop and implement more complex functions.  

"""
import os, subprocess
import numpy as np
from collections import OrderedDict

#length of a vector
def vectornorm(v1):
    return np.linalg.norm(v1)

#distance between two atoms     
def distance(v1, v2):
    return np.linalg.norm(v1-v2)    
    
#angle between two vectors in the unit of degree 
def angle(v1, v2):
    return np.arccos(np.dot(v1,v2)/(vectornorm(v1)*vectornorm(v2)))*(180/np.pi)

#A dihedral angle between the plane defined by points i, j and k and 
#the plane defined by points j, k and l   
def dihedral_angle(self, v1, v2, v3, v4):  
    n1 = unitnormvect(v1, v2, v3)
    n2 = unitnormvect(v2, v3, v4)
    return angle(n1, n2)


#the volume of a parallelepiped defined by three vectors 
def tripleproduct(v1, v2, v3):
    return np.dot(v1,np.cross(v2,v3))

#a vector normal to the plane defined by three points 
def unitnormvect(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    cp = np.cross(v1, v2)
    norm = np.linalg.norm(cp)
    return cp / norm 
    
#generating two orthogonal vectors that are orthogonal to a particular direction
def orthogonal(v):
    v = v/np.linalg.norm(v)
    x = np.random.randn(3)  # take a random vector
    x -= x.dot(v) * v       # make it orthogonal to k
    x /= np.linalg.norm(v)  # normalize it
    w = np.cross(v, x) 
    return x, w


#cf. http://tinyurl.com/ycmyx4b5 
def build_tmatrix(v1, v2, v3, operator):
        
    a = vectornorm(v1) #length of the v1 vector
    b = vectornorm(v2) #length of the v2 vector
    c = vectornorm(v3) #lenght of the v3 vector
    
    alpha = angle(v2,v3)*(np.pi/180) #The angle between v2 and v3 vectors
    beta  = angle(v1,v3)*(np.pi/180) #The angle between v1 and v3 vectors
    gamma = angle(v1,v2)*(np.pi/180) #The angle betweem v1 and v2  vectors
    
#    sa = np.sin(alpha)
#    sb = np.sin(beta)
    sg = np.sin(gamma)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
 
    #v is the volume of a unit parallelepiped             
    v =(a*b*c)*np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)
                          
    if operator == 'f2c':            
        tmatrix = np.array([
                             [a, b*cg, c*cb],
                             [0, b*sg, (c*(ca-cb*cg))/sg],
                             [0, 0, v/(a*b*sg)]])            
        return tmatrix             
    elif operator == 'c2f':        
        tmatrix = np.array([
        [1/a, -(cg)/(a*sg), ((b*cg*c*(ca-cb*cg)/sg)-b*c*cb*sg)*(1/v)],
        [0, 1/(b*sg), -(a*c*(ca-cb*cg))/(v*sg)],
        [0, 0, (a*b*sg)/v]])          
        return tmatrix
    else:
        print('Please assign the matrix type: f2c or c2f')

#building rotation matrix for rotating vectors by an angle theta about 
#the x-, y-, or z-axis. 
#cf. https://en.wikipedia.org/wiki/Rotation_matrix
def build_rmatrix(theta=0.00, axis = 'z'):
    theta = theta*(np.pi/180) #Unit in radian
    st = np.sin(theta) #sin(theta)
    ct = np.cos(theta) #cos(theta)
    if axis == 'z' or axis == 'Z':
        rmatrix = np.array([[ct, -st, 0.0],
                            [st, ct, 0.0],
                            [0.0, 0.0, 1.0]])
    elif axis =='y' or axis == 'Y':
        rmatrix = np.array([[ct, 0.0, st],
                            [0.0, 1.0, 0.0],
                            [-st, 0.0, ct]])
    elif axis == 'x' or axis == 'X':
        rmatrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, ct, -st],
                            [0.0, st, ct]])
    else:
        rmatrix = np.identity(3) #identity matrix
        print ('Please assign a reasonable axis')
    return rmatrix

#building euler rotation matrix
def build_eulermatrix(phi=0.00, theta=0.00, psi=0.00):
    """cf. http://mathworld.wolfram.com/EulerAngles.html
    
    phi :
            The 1st rotation angle around the z axis.
    theta :
            Rotation around the x axis.
    psi :
            2nd rotation around the z axis.
            
    """
    phi = phi*(np.pi/180)
    theta = theta*(np.pi/180)
    psi = psi*(np.pi/180)
    
    #the first rotation is by an angle phi about the z-axis 
    D = np.array([[np.cos(phi), np.sin(phi), 0.],
                  [-np.sin(phi), np.cos(phi), 0.],
                  [0.0, 0.0, 1.0]])
    
    #the second rotation is by an angle theta in [0,pi] about the former x-axis 
    C = np.array([[1.0, 0.0, 0.0],
                  [0.00, np.cos(theta), np.sin(theta)],
                  [0.00, -np.sin(theta), np.cos(theta)]])
    
    # the third rotation is by an angle psi about the former z-axis 
    B = np.array([[np.cos(psi), np.sin(psi), 0.],
                  [-np.sin(psi), np.cos(psi), 0.],
                  [0.0, 0.0, 1.0]])
    
    # Eular matrix
    matrix = np.dot(B, np.dot(C, D))
    return matrix

#Get redundant internal coordinates and return values in ap.arrays
def build_rc(positions):
    N = len(positions)
    distances  = [ ]
    angles  = [ ]
    dhangles = [ ]
    for i, pi in enumerate(positions):
        for j, pj in enumerate(positions):
            if i < j and N >= 2:
                value = distance(pi , pj)
                distances.append(value)
            for k, pk in enumerate(positions):
                if i < j and j < k and i < k and N >= 3:
                    value = angle(pi ,pj, pk)
                    angles.append(value)
                for l, pl in enumerate(positions):
                    if i < j and i < k and i < l and \
                       j < k and j < l and k < l and N >= 4:
                           value = dihedral_angle(pi, pj, pk, pl)
                           dhangles.append(value)                     
    return np.array(distances), np.array(angles), np.array(dhangles)


#Get redundant internal coordinates and return values in a dict. 
def build_rc_in_dict(positions):
    N = len(positions)
    distances  = OrderedDict()
    angles  = OrderedDict()
    dhangles = OrderedDict()
    for i, pi in enumerate(positions):
        for j, pj in enumerate(positions):
            if i < j and N >= 2:
                key = 'R(%s, %s)'%(i, j)
                value = distance(pi , pj)
                distances[key] = value
            for k, pk in enumerate(positions):
                if i < j and j < k and i < k and N >= 3:
                    key = 'A(%s, %s, %s)' %(i, j, k)
                    value = angle(pi ,pj, pk)
                    angles[key] = value
                for l, pl in enumerate(positions):
                    if i < j and i < k and i < l and \
                       j < k and j < l and k < l and N >= 4:
                           key = 'D(%s, %s, %s, %s)' %(i, j, k, l)
                           value = dihedral_angle(pi, pj, pk, pl)
                           dhangles[key] = value
    ric = OrderedDict([('distances',distances), ('angles', angles), ('dihedral', dhangles)])                       
    return ric

#distance between two clusters      
def clus_distance(posmtx1, posmtx2, mode=None):
    if mode == 'c':
        ct1 = get_center_point(posmtx1)
        ct2 = get_center_point(posmtx2)
        d = np.linalg.norm(ct2-ct1)
    else:
        dmatrix = get_distance_matrix(posmtx1, posmtx2)
        d = np.min(dmatrix)
    return d   
    
#Rotate a set of position vectors along z, y or x axis counterclockwise
#through an angle theta
def rotate_structure(posvects, theta, axis = 'z'):
    rmatrix = build_rmatrix(theta, axis) #unit in degree
    structure = np.dot(posvects, rmatrix.T)
    return structure

#Rotate a set of position vectors along z, y or x axis counterclockwise
#through an angle theta
def euler_rotate(posvects, phi=0.00, theta=0.00, psi=0.00):
    """phi, theta, psi => unit in degree"""
    ematrix = build_eulermatrix(phi, theta, psi) 
    structure = np.dot(posvects, ematrix.T)
    return structure

#Line–sphere intersection
#cf. http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
def distance_from_a_point_to_a_line(m0, p1, p0=np.array([0.00, 0.00, 0.00])):
    v = p1 - p0 #directing vector of the line
    q = p0 - m0  #the vector from the point p0 to the point m0  
    r = np.cross(v, q)
    d = np.linalg.norm(r) / np.linalg.norm(v)
    return d

#Line–sphere intersection
#cf. http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
def intxn_of_a_line_and_a_sphere(pt2, pt1=np.array([0.00, 0.00, 0.00]),
                            cent=np.array([0.00, 0.00, 0.00]), radius=1.00):
    
    v = pt2 - pt1 #a vector from a point pt1 to point pt2
    q = cent - pt1 #a vector from the sphere centerto point pt1
    
    a = np.dot(v, v)
    b = -2*np.dot(v, q)
    c = np.dot(q, q) - radius**2
    d = b**2 - 4*a*c   
    
    if d >= 0:
        t1 = (( -b + np.sqrt(d) ) / (2*a))
        t2 = (( -b - np.sqrt(d) ) / (2*a)) 
        ixn1 = pt1 + v * t1 #First intersection point
        ixn2 = pt1 + v * t2 #Second intersection point
    else:
        ixn1 = None
        ixn2 = None
        
    return ixn1, ixn2

#The intersection plane of two spheres
#cf. http://www.ambrsoft.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm
def intxn_of_two_spheres(x1, x2, r1, r2):
    #the normal vector of the intersection plane
    dvec = x2 - x1
    d = vectornorm(dvec)
    n = dvec /d
    
    #the radius of the intersection circle
    h = np.sqrt ( 4* (r1**2) * (d**2) - (r1**2 + d**2 - r2**2)**2 )/ (2*d)
    
    #the center of the intersection circl
    g = np.sqrt(r2**2 - h**2)
    c = x2 - g*n
    
    return h, c, n  

#The projection of a point to a plane
def projection_of_a_point_to_a_plane(x0, p0=[0, 0, 0], n=[1, 0, 0]):
    v = x0 - p0
    n = n / vectornorm(n) #the normal vector of the plane
    d = np.dot(v, n)
    p1 = x0 - (d * n)
    return p1

#Move a set of position vectors along a given direction
def move_structure(posvects, v1):
    structure = np.array(posvects) + np.array(v1) 
    return structure

#Move a set of dictionarian coordinates to along a given direction
def move_structure2(coordinates, v1):
    for element in coordinates.keys():
        coordinates[element] = coordinates[element] + v1
    return coordinates

#Get center point 
def get_center_point(posvects):
    center = np.array(posvects).sum(0)/len(posvects)
    return center

#Get center point of a dictionarian coordinate 
def get_center_point2(coordinates):
    entirety = merge(coordinates)
    center = np.array(entirety).sum(0)/len(entirety)
    return center
    
#Set the center of a structure to (0.00, 0.00, 0.00).        
def move_to_origin(posvects):
    center = get_center_point(posvects)
    structure = posvects - center
    return structure

#Set the center of a set of dictionarian coordinates to (0.00, 0.00, 0.00).     
def move_to_origin2(coordinates):
    center = get_center_point2(coordinates)
    coordinates = move_structure2(coordinates, center)
    return coordinates

#Move the center of a structure to a specific point      
def move_to_the_point(structure, point):
    center = get_center_point(structure)
    movect = point - center
    structure = move_structure(structure, movect)
    return structure

#Move the center of a set of dictionarian coordinates to a specific point 
#return a new dictionarian coordinate    
def move_to_the_point2(coordinates, point):
    entirety = merge(coordinates)
    center = get_center_point(entirety)
    movect = point - center
    for element in coordinates.keys():
        coordinates[element] = coordinates[element] + movect
    return coordinates

#Give a set of dictionarian coordinates and return a set of coordinates 
#in a single np.array
def merge(coordinates):
    entirety = np.ndarray(shape=(0,3), dtype=float)
    for element in coordinates.keys():
        entirety = np.concatenate((entirety, coordinates[element]))
    return entirety

#Combine two sets of dictionarian coordinates into one dictionary
def combine(dict1, dict2):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    combination = OrderedDict()
    for element in keys1:
        if element in keys2: #co-elements in dict1 and dict2
            combination[element] = np.concatenate((dict1[element], dict2[element]))
            keys2.remove(element)
        else:#elements only in dict1
            combination[element]  = dict1[element]
    for element in keys2: #elements only in dict2
        combination[element]  = dict2[element]
        
    return combination

#match a structure according to a set of items in dictionary and 
#return a set of dictionarian coordinates
def match(structure, dictitem):
    coordinates = OrderedDict()
    for element in dictitem.keys():
        natoms = dictitem[element]
        coordinates[element] = structure[0:natoms] 
        structure = structure[natoms:]
    return coordinates
       
#Truncate a part of coordinates from a set of dictionarian coordinates according to 
#an order dictionary 
def truncate(coordinates, dictitem):
    #dictitem   = OrderedDict(dictitem)
    truncation = OrderedDict()
    for element in coordinates.keys():
        if element in dictitem.keys():
            ntruns = len(coordinates[element]) - dictitem[element] 
            truncation[element] = coordinates[element][ntruns:]
    return truncation

#pair keys and number of values of a dictionary and return a dictionary
def pair_key_and_amount(coordinates):
    pairdict = OrderedDict()
    for element in coordinates.keys():
        pairdict[element] = len(coordinates[element])
    return pairdict
        
#Sort a set of position vectors according to their vectornorm
def sort_vectors(posvects):
    """
    lengths = []; vectors = []
    for vector in posvects:
        vectors.append(vector)
        lengths.append(np.linalg.norm(vector))
    pairs = sorted(zip(lengths,vectors ))
    vectors = [vector for length, vector in pairs]
    vectors= np.array(vectors)
    """
    indexlist = np.argsort (np.linalg.norm(posvects,axis=1)) 
    #posvects = posvects[indexlist]
    return posvects[indexlist]

#Generate a unit normal vector randomly  
def generate_a_normvect(perpendicular=''):
    vect = np.random.uniform(-1, 1, (3,))
    if perpendicular.upper() == 'XY':
        vect[2] = 0.00
    elif perpendicular.upper() == 'XZ':
        vect[1] = 0.00
    elif perpendicular.upper() == 'YZ':
        vect[0] = 0.00
    else:
        vect = vect
    norm = np.linalg.norm(vect)
    return vect / norm

"""
#Select k elements from a given set S of n elements randomly 
def selector(k, n):
    k = int(k); n = int(n)
    selected = [ ]
    while len(selected) != k:
        index = np.random.randint(0, n)
        if index not in selected:
            selected.append(index)
    if len(selected) == 1:
        selected = selected[0]
    return selected
"""

#Select k elements from a given set S of n elements randomly 
def selector(k, n):
    select = np.random.choice(n, k, replace=False) 
    return select
    
 

#Select k elements from a given set S of n elements randomly 
#according a set of values as reference
def selector2(k, n, refvalues): 
    k = int(k); n = int(n)
    selected = [ ]
    while True:
        i = np.random.randint(0, n)
        p = np.random.rand()
        f = refvalues[i] 
        if f > p and i not in selected:
            selected.append(i) 
        if len(selected) == k:
            break   

    if len(selected) == 1:
        selected = selected[0]
    return selected   


#Sort two lists and use list1 as the reference 
def sort_two_lists(list1, list2):
    zipped_lists = zip(list1, list2)
    sorted_pairs = sorted(zipped_lists, key=lambda x: x[0])
    tuples = zip(*sorted_pairs)
    list1, list2 = [ list(tuple) for tuple in  tuples]    
    return list1, list2

#Convert a set of coordinates in a dictionary to a single np.array 
def concatenate_dict_values(dictionary):
    keys = list(dictionary.keys())
    values = [dictionary[key] for key in keys]
    return np.concatenate((values))

#Filter all documents in a directory and return only directories
def dirfilter(path):
    directories = [i for i in sorted(os.listdir(path)) if os.path.isdir('%s/%s' %(path,i))] 
    return directories

#Filter all documents in a directory and return only files
def filefilter(path):
    files = [i for i in sorted(os.listdir(path)) if os.path.isfile('%s/%s' %(path,i))] 
    return files


#---------------Calculating the distances between the atoms-------------------
#This is a matrix with a dimension of Natoms by Natoms.
#For each element in the rows of the matrix, it corresponds to the bond 
#lengths of the atom_i and i != j
def get_distance_matrix(XA, XB=None, replacement=True):
    if XB is None:
        XB = XA
    
    XA = np.array(XA);  XB = np.array(XB)
    from scipy.spatial.distance import cdist 
    dmatrix = cdist(XA, XB, metric='euclidean')

    if replacement and np.all(XA == XB):
        for i in range(len(dmatrix)):
            dmatrix[i,i] = np.nan

    return dmatrix

"""
def get_distance_matrix(XA):

    natoms = len(XA)
    
    #a zero matrix in a dimension of natoms by natoms
    dmatrix=np.zeros((natoms, natoms))
    
    for atom_i, pos_i in enumerate(XA):
        for atom_j, pos_j in enumerate(XA):
            if atom_i != atom_j:
                #The distance between atom_i and atom_j
                distance =np.linalg.norm(pos_i - pos_j) 
                dmatrix[atom_i][atom_j] = distance 
            else:
                #If atom_i == atmo_j, then set the distance value as nan.
                dmatrix[atom_i][atom_j] = None 
"""


def get_distances_betwen_two_clus(clu1, clu2):
    cent1 = get_center_point(clu1)
    cent2 = get_center_point(clu2)
    dmatrix = get_distance_matrix(clu1, clu2)
    dmin  = np.min(dmatrix)
    dmax  = np.max(dmatrix)
    dcent = np.linalg.norm(cent2 - cent1)
    return dmin, dmax, dcent



#----------------------Get fragments from a random structure------------------
#After generating a randomly structure, atoms are possible far away each other.
#This part is to find the atoms within a certain distance and treat them as a fragment         
def get_fragments(dmatrix, cutoff_up):
    natoms  = len(dmatrix)
    #This is a list of atoms which are not assigned to any part. 
    atoms_left = list(range(natoms)) 
    fragments = [ ]
    while(atoms_left != [ ]): 
        fragment = [atoms_left[0]]
        switch = 'on'
        while(switch != 'off'):
            for atom_i in fragment: #0
                for atom_j in atoms_left:
                    if atom_i !=atom_j:#0 1, 0 2, 0 3, 0 4
                        distance = dmatrix[atom_i][atom_j]
                        #if the distance between i and j is smaller than cutoff_up,
                        #they are in a same fragment.
                        if distance < cutoff_up and atom_j not in fragment:  
                            fragment.append(atom_j) #0, 1, 2
            #Atoms in fragment means that they have been assigned to be a part                
            for atom_i in fragment: 
                if atom_i in atoms_left:
                    atoms_left.remove(atom_i)  #Because atom_i has been assigned to be a fragment, it was removed from atom_left
            switch = 'off'
            fragments.append(fragment) # now we get a set of fragments: [[i, j ,k,l], [m,n], o, p, [q,r,s,t], [u,v].....]
    return fragments

#-----------------------------Find the largest fragment ----------------------
#This function is to find the fragment with most number of atoms, called the main fragment, and     
#the other fragments will be connected to the main fragment with a designed distance     
def get_main_fragment_index(fragments):
    mainfrag_index = -1.00; 
    mainfrag_len = -1.00; 
    #mainfrag_atoms = [ ]
    for frag_i, fragment in enumerate(fragments):
        if len(fragment) >  mainfrag_len:
             mainfrag_len = len(fragment)
             mainfrag_index = frag_i
    return mainfrag_index

#-----------------Find the shortest distance between two fragments-------------
#Atom_i in the mainfrag and Atom_j in the minorfrag will determine the shortest distance between two fragments    
#The vector of (Atom_i - Atom_j) will be used as a moving vector for bridging fragments. 
def get_shortest_of_two_fragments(dmatrix, fragment1, fragment2):
    if len(fragment1) >= len(fragment2):
        mainfrag = fragment1
        minorfrag = fragment2
    else:
        mainfrag = fragment2
        minorfrag = fragment1        
    shortest = 1e23; main = 0; minor = 0       
    for atom_i in mainfrag:
        for atom_j in minorfrag:
            distance = dmatrix[atom_i][atom_j]
            if distance < shortest:
                main = atom_i; minor = atom_j #atom_i in the mainfrag and atom_j in the minorfrag     
                shortest = distance #The shortest distance between two fragments        
    return main, minor, shortest

#------------------Bridge two fragments with a designed distance---------------
#This function is to bridge a fragment with the main fragment
#After bridging, the two fragments will have a shortest distance of 'bridge'               
def move_fragments(structure, bridge, cutoff_up):
    dmatrix = get_distance_matrix(structure)
    fragments = get_fragments(dmatrix, cutoff_up) #fragments: [[i, j ,k,l], [m,n], o, p, [q,r,s,t], [u,v].....]
    nfrags = len(fragments)
    frag_indexes = list(range(nfrags))
    if nfrags == 1:
        return structure
    else:
        mainfrag_index = get_main_fragment_index(fragments) #index, [q,r,s,t]
        frag_indexes.remove(mainfrag_index)
        for frag_index in frag_indexes:
            main, minor, shortest = get_shortest_of_two_fragments(dmatrix, fragments[mainfrag_index], fragments[frag_index])
            move_vector = structure[main] - structure[minor]
            move_vector = move_vector*(shortest - bridge) /shortest
            for atom_move in fragments[frag_index]:
                structure[atom_move] =  structure[atom_move] + move_vector
                #new_distance = np.linalg.norm(structure[main]-structure[atom_move]) #np.sqrt(((structure[main]-structure[atom_move])**2).sum())
        return structure


#-------------------------Find a set of points as anchor ---------------------
#This function will try to find a set of points in a cluster, which will directly
#contact with a surface. Then The cluster can be anchored in the surface via 
#these points.
def search_anchoring_points(structure, nanchors=3, tolerance=0.15, maxattempts=100000): 
    for i in range(maxattempts):
        rotaxis = np.random.choice(['x', 'y', 'z'])
        angle = np.random.randint(0,360)
        structure = rotate_structure(structure, angle, rotaxis)
        blockmatrix = structure * np.array([0,0,1])
        dmatrix = get_distance_matrix(blockmatrix)
        
        #The distances in the z-direction between atoms within tolerance value,
        #will be treated as co-plane. 
        fragments = [fragment for fragment in get_fragments(dmatrix, tolerance)\
                     if nanchors == len(fragment) ]
        
        nfragments = len(fragments)
        if nfragments != 0:
            index = np.random.randint(0, nfragments)
            fragment = fragments[index]
            candidate =np.array([structure[i] for i in fragment])
        else:
            continue
        
        candminiz = min([position[2] for position in candidate]) 
        struminiz = min([position[2] for position in structure]) 
        if candminiz <= struminiz:
            anchors = candidate
            indexes = fragment
            break
        else:
            continue        
    return indexes, anchors, structure
       
#Give a string, this function will search the lines in a file that match 
#the string. This is only work for unix-based systems!!  
def grep_a_string(string, filename, tail='-1'):
    cmd =  'grep "%s" %s | tail %s'  %(string, filename, tail)   
    stdout, stderr = execute(cmd)
    #line = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    #info = line.stdout.read().strip().decode('utf-8').split() 
    return stdout.decode('utf-8').split()

#Print strings to a file
def print_to_file(strings, filename, mode='a', sep='\n'):
    f = open(filename, mode)
    if mode == 'a':
        f.write('\n') #Insert a seperation line
    if isinstance(strings, str):
        string = strings
        f.write(string.strip())
    else:
        for string in strings:
            f.write(string.strip())
            f.write(sep)
    f.close()

#check a string whether a number:
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False 
    
#Execute Shell Commands and return stdout and stderr
def execute(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr