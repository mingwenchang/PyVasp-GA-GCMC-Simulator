#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAME
        monitor.py -  A central module for contronling 
                      and monitoring parall GA/GCMC calculations

                           
DESCRIPTION
        Manage jobs in a hpc qms. 
        Creat working folders for GA/GCMC calculations
        Collect and analysis simulation results.  
           
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import os, sys, shutil, time, yaml
import numpy as np
import src.modules.assister as ast
#import modules.vasp_io2 as vio
import src.modules.examiner as exm
import src.modules.constructor as con
from src.modules.units import kb
from copy import copy
#from collections import OrderedDict

class QueueManager:
    def __init__(self, jobname=None, workdir=None, ncpus=None, walltime=None, manifest=None):
        
        """Class for representing a queue manager.
    
        Parameters:
    
        jobname: str or int
            job name 
            
        workdir: str
            the path to the work folder 
        
        ncpus: int
            request slot range for parallel jobs
            
        walltime: str (HH:MM:SS)
            maximum running time for a job
            
        manifest: str 
             a path to a yml file, which includes
            
                 1. a script template for the current queue manager system (qms)
                 2. a qms command for submitting jobs
                 3. a qms command for killing jobs
                 4. a qms command for checking jobs status
                 5. an index value for navigating jobid
                 6. an index for navigating job status
                    
        """
        
        if jobname is None:
            self.jobname = "job"
        else:
            self.jobname = jobname
        
        if workdir is None:
            self.workdir = os.getcwd()
        else:
            self.workdir = workdir 
            
        if ncpus is None:
            self.ncpus = 1
        else:
            self.ncpus = ncpus

        if isinstance(walltime, str) or walltime is None:
            self.walltime = "04:00:00" #four hours
        else: #convert hr to sec
            self.walltime = time.strftime("%H:%M:%S", time.gmtime(walltime * 3600 )) 
         
            
        cwd = os.path.dirname(__file__)      
        scripts = { None: os.path.join(cwd, '../manifest/cartesius.yml'),
                   'cartesius':  os.path.join(cwd, '../manifest/cartesius.yml'),
                   'cartesius-gamma':  os.path.join(cwd, '../manifest/cartesius-gamma.yml'),
                   'cartesius-short': os.path.join(cwd, '../manifest/cartesius-short.yml'),
                   'cartesius-short-gamma': os.path.join(cwd, '../manifest/cartesius-short-gamma.yml'),
                   'tue': os.path.join(cwd, '../manifest/tue.yml') 
                 }     
        try:
            if "short" in manifest:
                self.walltime = "01:00:00" #1.00 hours
            self.manifest = yaml.safe_load(open(scripts[manifest], 'rb'))
        except:
            if os.path.exists(manifest):
                self.manifest = yaml.safe_load(open(manifest, 'rb'))
            else:
                raise FileNotFoundError
    
        self.status = None
        self.jobid = None
        self.runfile = None
        self.jobinfo = None
        self.ntrips = 0
        
    def __repr__(self):
        return "Queue(%s)" %(self.jobname)
            
    #write a job runfile 
    def writejobfile(self, kwargs=None):            
        if kwargs is None:
            kwargs = {}
            kwargs["JOBNAME"] = self.jobname
            kwargs["WORKDIR"] = self.workdir
            kwargs["NCPUS"] = self.ncpus
            kwargs["WALLTIME"] = self.walltime
        self.runfile = "%s/%s.run" %(self.workdir, kwargs["JOBNAME"]) 
        with open(self.runfile, 'w') as txt:
            txt.write(self.manifest["script"] %kwargs)
        return self.runfile 
    
    def writejobinfo(self, jobinfo="jobinfo"):
        self.jobinfo = os.path.join(self.workdir, jobinfo)
        date = time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime())
        with open( self.jobinfo, 'w') as txt:
            txt.write(date + '\n')
            txt.write("num. of trips: %s\n" %self.ntrips)
            txt.write("Job Name: %s\n" %self.jobname)
            txt.write("Job ID: %s\n" %self.jobid)
            if self.status == 'killed':
                txt.write("Job status: %s\n" %self.status)
            else:
                txt.write("Job status: %s\n" %self.checkjob())
    
    #Submit a Job to the queue
    def submitjob(self):
        cmd = self.manifest["submit"] + ' < ' + self.runfile
        idx = self.manifest["jobid-idx"] 
        stdout, stderr = ast.execute(cmd)
        self.jobid = self.__parse_jobid(stdout, stderr, idx)
        return self.jobid 
    
    #Check a job's status according to a jobID and return the job status
    def checkjob(self):
        cmd = self.manifest["check"]  + ' ' +  str(self.jobid)
        idx = self.manifest["status-idx"]
        stdout, stderr = ast.execute(cmd)
        self.status  = self.__parse_jobstatus(stdout, stderr, idx)
        return self.status
    
    #Kill a Job according to a jobID   
    def killjob(self):
        cmd = self.manifest["kill"]  + ' ' +  str(self.jobid)
        stdout, stderr = ast.execute(cmd)
        self.status  = 'killed'
        return self.status
    
    def __parse_jobstatus(self, stdout, stderr, idx):
        #Possible outputs:
        running = ["R", "RUN", "RUNNING", "T"]
        queuing = ["PD", "Q", "H", "W"]
        
        try:
            string = stdout.decode().split()[idx].upper()
        except IndexError:
            #print ("Error from the qms:")
            #print (stdout.decode())
            string = None
    
        if string in running:
            return "running"
        elif string in queuing:
            return "queuing"
        else:
            return "done"

    def __parse_jobid(self, stdout, stderr, idx):
        try:
            jobid = int(stdout.decode().split()[idx]) 
        except:
            #print ("Error from the qms:")
            #print (stdout.decode())
            jobid = None
        return jobid
 

class GAFileManager:
    """Class for representing a GA FileManager.
    
    1. buid key folders  
    
    2. buid working forlders

    """    
    def __init__(self, inputs=["INCAR", "KPOINTS", "POTCAR", "POSCAR"]):
        
        self.rootdir   = os.getcwd() 
        self.pooldir   = os.path.join(self.rootdir, 'pool') #Candidates are collected here
        self.offsprdir = os.path.join(self.rootdir, 'offspring') #Workding dir for running vasp jobs
        self.restdir   = os.path.join(self.rootdir, 'restroom') #Temporary dir to store successful vasp jobs
        
        self.inputs = {}
        for term in inputs:
            self.inputs[term] = os.path.join(self.rootdir, term) 
        
    def initialize(self):
        print("#Initialization at %s" %(time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime())))
        print ("Checking the necessary input files for GA")
        
        missing = [file for file in self.inputs if not  os.path.exists(self.inputs[file])]     
        if len(missing) > 0:
            print("The following files are missing: %s" %(" ".join(missing)))
            raise FileNotFoundError
        
        for folder in (self.pooldir, self.offsprdir, self.restdir):
            if not os.path.isdir(folder):
                os.mkdir(folder)
                print ("Creating the %s folder" %(os.path.basename(folder)))
                
    @property
    def ncandiates(self):
        return len(ast.dirfilter(self.pooldir))

    @property
    def nthgen(self):
        return len(ast.dirfilter(self.offsprdir))
    
    @property
    def nstrus(self):
        return len(ast.dirfilter(self.gendir))  
    
    def mkgendir(self, tag):
        self.gendir = os.path.join(self.offsprdir, tag)
        os.mkdir(self.gendir) 
        return self.gendir

    def prepoptdir(self, tag):
        self.optdir = os.path.join(self.gendir, tag)
        os.mkdir(self.optdir) 
        for file in self.inputs:
            shutil.copy(self.inputs[file], self.optdir)
        return self.optdir
    
    def amalgamate(self, ResultParserInstance, StruExaminerInstance, tag="cand_"):
        if len(ast.dirfilter(self.pooldir)) == 0:
            optdir = os.path.join(self.restdir, ast.dirfilter(self.restdir)[0])  
            newdir = os.path.join(self.pooldir, tag + str(1).zfill(3))
            shutil.move(optdir, newdir)
        
        for optdir in ast.dirfilter(self.restdir) :
            for candir in ast.dirfilter(self.pooldir):
                optdir = os.path.join(self.restdir, optdir)  
                candir = os.path.join(self.pooldir, candir)  

                optresult  = copy(ResultParserInstance)
                candresult = copy(ResultParserInstance)
                optresult.workdir = optdir 
                candresult.workdir = candir
                
                identical = StruExaminerInstance.is_identical(optresult, candresult)
 
                if identical:
                    print ('Found duplicate structures at: \n%s\n%s' %(optdir, candir))
                    if optresult.energy < candresult.energy: 
                        print ('Replace: ', candir)
                        print ('By: ', optdir)
                        shutil.rmtree(candir)
                        shutil.move(optdir, candir)
                    else:
                        print ('Preserve:\n', candir)
                        print ('Abandon:\n', optdir)
                        shutil.rmtree(optdir)
                    break
            else:
                ncands = len(ast.dirfilter(self.pooldir))
                newdir = os.path.join(self.pooldir, tag + str(ncands+1).zfill(3))
                print ("move: ", optdir)
                print ("===> ", newdir)
                shutil.move(optdir, newdir)  

class GAParametersManager:
    def __init__(self, gaini=None):
        
        #inputfile setting:
        self.inputs = ["INCAR", "KPOINTS", "POTCAR", "POSCAR"]
        
        #environment setting:
        self.vacuum = True 
        self.temperature = 300.
        self.pressure = 1.0
        self.gasphase= None
        self.chempot = None
        
        #GA run setting:
        self.gamode = "free" #free/supported
        self.maxgacycle =  30 # maximum itetative cycle for GA 
        self.genpopsize = 12 #the population size for GA per generation
        self.mutationrate = 0.80 # 0 < mutation rate < 1
        self.idtrate = [0.333, 0.333, 0.333]
         
        #HPC setting
        self.hpc = 'cartesius' #cartesius/tue/(a path to a yaml file) 
        self.ncpus = 24 #Request number of processors (cpus) per a job 
        self.walltime = 4 #the time limit (hours) for a job 
        self.refreshtime  = 900 #refresh time (sec) for the info update of GA 
        self.maxretry = 2 #max. attempts for an unconverged job
        self.naptime = 600
        self.epsilon = 1.3 #control how many parallel jobs (=epsilon*genpopsize) at the same time 
        
        #Converge criterion setting
        self.force = 0.05  #force unit in Angstrom/eV (same as the ediffg value in INCAR)
        
        #Cluster landing setting:
        self.landingat = np.array([0.50, 0.50, 0.50])
        self.altitude  = 2.30
        self.nanchors = 3
            
        if gaini is not None:
            self.read_manifest(gaini)

    def read_manifest(self, inputfile=None):
        with open(inputfile, 'rb') as f:
            params = yaml.safe_load(f)
            for key in params:
                setattr(self, key, params[key]) 
                
        self._parse_cluster()
        self._parse_inputs()
        
        if not self.vacuum: 
            self._parse_gasphase() 
            
    def _parse_cluster(self):
        atomtypes = []; natoms = []  
        for term in self.cluster:
            atom = list(term.keys())[0]
            num = list(term.values())[0]
            atomtypes.append(atom)
            natoms.append(num)
        self.cluster = con.DummyCluster(atomtypes,natoms)
        
    def _parse_inputs(self):
        if self.inputs == 'vasp':
            self.inputs = ["INCAR", "KPOINTS", "POTCAR", "POSCAR"]
        elif self.inputs == 'vasp-vdw':
            self.inputs = ["INCAR", "KPOINTS", "POTCAR", "POSCAR", "vvdw_kernel.bindat"]  
     
    def _parse_gasphase(self):
        self.gasphase = self.gasphase.split(',')
        if len(self.gasphase) == 1:
            mol1 = self.gasphase[0].strip()
            key1 = 'X' + str(1)
            setattr(self, key1, mol1)
        else:
            mol1 = self.gasphase[0].strip()
            key1 = 'X' + str(1)
            setattr(self, key1, mol1)
            
            mol2 = self.gasphase[1].strip()
            key2 = 'X' + str(2)
            setattr(self, key2, mol2)
            
        chempot = { }
        for dicterm in self.chempot:
            chempot = {**chempot, **dicterm}
        self.chempot = chempot

    @property
    def nparallels(self):
        return int(self.genpopsize * self.epsilon) 
  

def snooze(naptime=600, maxtime=24*3600):
    def decorator(func):
        def wrapper(*args, **kwagrs): 
            start = time.time()
            job = func(*args, **kwagrs)
            while True:
                now  = time.time()
                if job.jobid is None:
                    if now - start < maxtime:
                        subtime = time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime(now + naptime))
                        print ("Captured an error when submitting a job!")
                        print ("Probably, the jobs limit has been reached, or the qms is in busy") 
                        print ("The %s job will be re-submited at %s" %(job.jobname, subtime))
                        time.sleep(naptime)
                    else: 
                        print ('Maximum waiting time has been reached.')
                        print ('GA has been automatically terminated.')
                        sys.exit()
                else:
                    print ('The %s job has been submitted to the Queue management system (qms)' \
                           %(job.jobname))  
                    job.writejobinfo()
                    break
                job.submitjob()
            return job
        return wrapper
    return decorator

 
class GAParallelController(GAFileManager):
    """Class for representing a GA Parallel Controller.
    
    Features:
    
        1. submit multiple jobs
        2. monitor parallel jobs 
        3. automatically collect results 
        
    Parameters
        GAParameterManagerInstance: cls-inst
        ResultParserInstance: cls-inst
        StruExaminerInstance: cls-inst

    """    
    
    def __init__(self, GAParameterManagerInstance=None,
                       ResultAnalyzerInstance=None,
                       StruExaminerInstance=None):
        
        self.gaparmgr  = GAParameterManagerInstance
        self.rtanalyzer  = ResultAnalyzerInstance
        self.struexamr = StruExaminerInstance
        super().__init__(self.gaparmgr.inputs) 
        
    def initialize(self):
        super().initialize()
        if  os.path.exists(self.restdir) and len(ast.dirfilter(self.restdir)) > 0:
            print ('Analyzing the restroom folder.....')
            self.amalgamate(self.rtanalyzer, self.struexamr)
            
    @snooze()      
    def submitjob(self, jobdir):
        jobname = os.path.basename(jobdir)
        job = QueueManager(jobname=jobname, 
                           workdir=jobdir,
                           ncpus=self.gaparmgr.ncpus, 
                           walltime=self.gaparmgr.walltime, 
                           manifest=self.gaparmgr.hpc)
        job.writejobfile()
        #job.jobid = 0 
        job.submitjob()
        time.sleep(60)
        return job
        
    def submit_parallel_jobs(self, pardir=None):
        if pardir is None:
            pardir = self.gendir
            
        self.jobs = []
        for jobdir in ast.dirfilter(pardir):
            jobdir = os.path.join(pardir, jobdir) 
            self.jobs.append(self.submitjob(jobdir))
        return self.jobs
             
    def kill_parallel_jobs(self, jobs=None):
        if jobs is None:
            jobs = self.jobs
        
        for job in jobs:
            job.killjob()
            job.writejobinfo()

    def parse_parallel_jobs(self, jobs=None):
        if jobs is None:
            jobs = self.jobs
        
        running = []; converged = []; unconverged = []
        for job in jobs:
            status = job.checkjob()
            if status == "done":
                self.rtanalyzer.workdir = job.workdir
                if self.rtanalyzer.is_converged():
                    converged.append(job)
                else:
                    unconverged.append(job)
            else:
                running.append(job)

        return running, converged, unconverged
    
    def monitor(self):
        start = time.time()
        while True:
            print("\nUpdate at %s" %(time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime())))
            running, converged, unconverged = self.parse_parallel_jobs()
            
            for job in converged:
                print ('Find a successful calculation!')
                print ('Move %s ===> %s' %(job.workdir, self.restdir))
                shutil.move(job.workdir, self.restdir)               
            
            if self.gaparmgr.maxretry > 0:
                for job in unconverged:
                    ntrips = job.ntrips  
                    if ntrips <= self.gaparmgr.maxretry:
                        print('%s job is not well converged' %(job.jobname))
                        print('Resubmit it again')
                        self.rtanalyzer.workdir = job.workdir
                        self.rtanalyzer.set_restart()
                        job = self.submitjob(job.workdir)
                        job.ntrips  = ntrips + 1 
                        running.append(job)
                    else:
                        print ("Maximum attempts for the %s job have reached, but it is not converged" %(job.jobname))
                        print ("Giving up this job")
              
            self.jobs = running
            print('Queueing and running jobs:')
            print ('{:<10}{:<10}{:<10}{:<20}'.format("No.", "JobID", "ST.", "JobDir")) 
            for i, job in enumerate(self.jobs): 
                job.writejobinfo()
                print ('{:<10}{:<10}{:<10}{:<20}'.format(str(i+1).zfill(3), job.jobid, job.status, job.workdir))
                                
            nsuccess = len(ast.dirfilter(self.restdir))
            if nsuccess >= self.gaparmgr.genpopsize:
                print ('%s converged jobs have been successfully collected' %(nsuccess))
                print ('Terminate the rest of uncompleted jobs')
                print ('\n')
                self.kill_parallel_jobs(self.jobs)
                break
            elif len(self.jobs) + nsuccess < self.gaparmgr.genpopsize:
                print ('\n')
                print ('!!! WARNING WARNING WARNING  WARNING  WARNING  WARNING WARNING WARNING  WARNING  WARNING!!! ')
                print ('GA cannot collect enough %s converged jobs at the current iterative cycle' %(self.gaparmgr.genpopsize))
                print ('A new iterative cycle will start directly.')
                print ('This probably has a negative impact for searching global minimum....')
                print ('If you see this warning too many times, ')
                print ('You should consider performing more extra cycles after the current GA simulation ')
                print ('\n')
                self.kill_parallel_jobs(self.jobs)
                break
            else:
                time.sleep(self.gaparmgr.refreshtime)
        
        self.amalgamate(self.rtanalyzer, self.struexamr)
        usedtime = time.time() - start 
        print("The current cycle used %s  " %(time.strftime("%H:%M:%S", time.gmtime(usedtime))))
 
    
class GAResultAnalyzer:
    """Class for representing a GA Result Analyzer .
    
    Features:
    
        1. Analyze GA results
        2. Save GA results
        
    Parameters
    
        pooldir: str
        ResultParserInstance: cls-inst
 
    """       
     
    def  __init__(self, GAParameterManagerInstance=None, 
                        ResultAnalyzerInstance=None, 
                        fitnessfunc=exm.fitfun_tanh,  pooldir=None):
        
        
        if pooldir is None:
            self.pooldir  = os.path.join(os.getcwd(), "pool")
            
        else:
            self.pooldir = pooldir
 
        self.rtanalyzer = ResultAnalyzerInstance
        self.gaparmgr = GAParameterManagerInstance
        self.fitnessfunc = fitnessfunc
        
    def analyze(self):
        energies = []; candidates= []; 
        for candir in ast.dirfilter(self.pooldir):
            candir = os.path.join(self.pooldir, candir)
            cand  = copy(self.rtanalyzer)
            cand.workdir = candir
            energies.append(cand.energy)
            candidates.append(cand)
            
        energies, candidates = ast.sort_two_lists(energies, candidates)
        emin, emax =  candidates[0].energy, candidates[-1].energy
        
        #Boltzman population 
        beta = -1/(kb * self.gaparmgr.temperature)  
        B = lambda ene: np.exp( (ene - emin) *  beta ) 
        F = [B(ene) for ene in energies]
        Z = np.sum(F)
        probability = [f/Z for f in F]
        
        competitiveness  = []
        for i , cand in enumerate(candidates):
            ene=cand.energy
            cand.fitness = self.fitnessfunc(ene=ene, emin=emin, emax=emax)
            competitiveness.append(cand.fitness) 
            cand.propulation = probability[i]
            
        self.energies = energies 
        self.candidates = candidates
        self.competitiveness = competitiveness 
        self.probability = probability
        

    def save(self, dirname=os.getcwd()):
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        gmevol    = os.path.join(dirname, 'gmevolve.xyz') #A movie file 
        allstrus  = os.path.join(dirname, 'allstrus.xyz') #A movie file 
        gmstru    = os.path.join(dirname, 'gm.xyz')  #gm.xyz
 
        if os.path.exists(allstrus):
            os.unlink(allstrus)
        
        self.gm.write(filename=gmstru, format='xyz', mode='w')
        self.gm.write(filename=gmevol, format='xyz', mode='a')  
                    
        with open('ga.dat', mode='w') as gadat:
            gadat.write('{:<10}{:<18}{:<18}{:<21}{:<21}\n'.format('#Rank', 
                                                          'Total Energy', 
                                                          'Relative Energy',
                                                          'Prob.(%s K)' %(self.gaparmgr.temperature), 
                                                          'Path to Candidate'))
            for i, cand in enumerate(self.candidates):
                cand.write(filename=allstrus, format='xyz', mode='a')
                #vio.write_xyz(cand._atoms,filename=allstrus, mode='a')
                gadat.write('{:<10}{:< 18.8f}{:< 18.8f}{:< 18.8f}{:<21}\n'.format(i+1,
                                                                  cand.energy, 
                                                                  cand.energy-self.gm.energy,
                                                                  cand.propulation,
                                                                  cand.workdir))              
    @property
    def gm(self):
        return self.candidates[0]
        


class GCGAResultAnalyzer(GAResultAnalyzer):

    def  __init__(self, ResultAnalyzerInstance=None, GAParameterManagerInstance=None,
                        dimension = None,
                        fitnessfunc=exm.fitfun_tanh,  pooldir=None):
        
        super().__init__(ResultAnalyzerInstance, GAParameterManagerInstance,
                         fitnessfunc=exm.fitfun_tanh,  pooldir=None)
        
        self.dimension = None
    
 
    
    def analyze(self):

        free = []; energies = []; candidates= [];
        for candir in ast.dirfilter(self.pooldir):
            candir = os.path.join(self.pooldir, candir)
            cand  = copy(self.rtanalyzer)
            cand.workdir = candir
            
            cand.free =  cand.energy
            for atom, num in zip(self.gaparmgr.cluster.atomtypes, self.gaparmgr.cluster.natoms):
                cand.free -= num * self.gaparmgr.chempot[atom]
           
            if len(cand.atoms.atomtypes) > len(self.gaparmgr.cluster.atomtypes):
                if len(self.gaparmgr.X1) == 2:
                    x = self.gaparmgr.X1[0]
                    X1 = self.gaparmgr.X1
                    num = cand.atoms.datomtypes[x]
                    cand.free -= num * self.gaparmgr.chempot[X1]
                else:
                    x = self.gaparmgr.X1
                    num = cand.atoms.datomtypes[x]
                    cand.free -= num * self.gaparmgr.chempot[x]
                    
            if "slab" in self.gaparmgr.chempot.keys():
                cand.free -= num * self.gaparmgr.chempot['slab']

                
            energies.append(cand.energy)
            free.append(cand.free)
            candidates.append(cand)            
        
        free, candidates = ast.sort_two_lists(free, candidates) 
        emin, emax =  candidates[0].free, candidates[-1].free
        #Boltzman population 
        beta = -1/(kb * self.gaparmgr.temperature)  
        B = lambda ene: np.exp( (ene - emin) *  beta ) 
        F = [B(ene) for ene in free]
        Z = np.sum(F)
        probability = [f/Z for f in F]
        
        if self.dimension is None:
            self.dimension = len(candidates)
        
        competitiveness  = []
        for i , cand in enumerate(candidates[0: self.dimension]):
            ene=cand.free
            cand.fitness = self.fitnessfunc(ene=ene, emin=emin, emax=emax)
            competitiveness.append(cand.fitness) 
            cand.propulation = probability[i]
            
        self.free = free   
        self.energies = energies 
        self.candidates = candidates
        self.competitiveness = competitiveness 
        self.probability = probability
            
    def save(self, dirname=os.getcwd()):
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        gmevol    = os.path.join(dirname, 'gmevolve.xyz') #A movie file 
        allstrus  = os.path.join(dirname, 'allstrus.xyz') #A movie file 
        gmstru    = os.path.join(dirname, 'gm.xyz')  #gm.xyz
 
        if os.path.exists(allstrus):
            os.unlink(allstrus)
        
        self.gm.write(filename=gmstru, format='xyz', mode='w')
        self.gm.write(filename=gmevol, format='xyz', mode='a')  
                    
        with open('ga.dat', mode='w') as gadat:
            gadat.write('{:<10}{:<18}{:<18}{:<21}{:<21}\n'.format('#Rank', 
                                                          'Free Energy', 
                                                          'Relative Energy',
                                                          'Prob.(%s K)' %(self.gaparmgr.temperature), 
                                                          'Path to Candidate'))
            for i, cand in enumerate(self.candidates):
                cand.write(filename=allstrus, format='xyz', mode='a')
                #vio.write_xyz(cand._atoms,filename=allstrus, mode='a')
                gadat.write('{:<10}{:< 18.8f}{:< 18.8f}{:< 18.8f}{:<21}\n'.format(i+1,
                                                                  cand.free, 
                                                                  cand.free-self.gm.free,
                                                                  cand.propulation,
                                                                  cand.workdir))     


