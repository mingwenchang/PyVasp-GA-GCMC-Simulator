#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        launch.py 

                        
DESCRIPTION
        launch GA/GCMC calculations
        

DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""


import os, sys, yaml
import subprocess

if __name__ == '__main__':
    if len(sys.argv) == 3:
        inp = os.path.join(os.getcwd(), sys.argv[2])
        manifest = yaml.safe_load(open(inp, 'rb'))

        if manifest['gagcmcdir'] == 'cwd':
            gagcmcdir = os.path.join(os.getcwd(), 'gagcmc')
        else:
            gagcmcdir = manifest['gagcmcdir'] 
    else:
        print ("Syntax:")
        print ("launch.py -[ga/gcmc] -[ga.inp/gcmc.inp]")
        print ("e.g. launch.py -ga ga.inp")
        print ("e.g. launch.py -gcmc gcmc.inp")
        raise SyntaxError 

    if sys.argv[1] ==  '-ga':
        mode = 'GA'
        if  manifest['vacuum']:    
            if manifest['gamode'] == 'free':
                gascript  = os.path.join(gagcmcdir, 'src/ga-free.py')
            elif manifest['gamode'] == 'supported':
                gascript  = os.path.join(gagcmcdir, 'src/ga-supported.py')
        else:
            if manifest['gamode'] == 'free':
                gascript  = os.path.join(gagcmcdir, 'src/gcga-free.py')
            elif manifest['gamode'] == 'supported':
                gascript  = os.path.join(gagcmcdir, 'src/gcga-supported.py')    
        cmd = 'nohup python3 -u %s %s >> ga.log 2>&1 & echo $!' %(gascript, inp)
    elif sys.argv[1] ==  '-gcmc':
        mode = 'GCMC'
        if manifest['gcmcmode'] == 'free':
            gcmcscript  = os.path.join(gagcmcdir, 'src/gcmc-free.py')
        elif manifest['gcmcmode'] == 'supported':
            gcmcscript  = os.path.join(gagcmcdir, 'src/gcmc-supported.py')
        cmd = 'nohup python3 -u %s %s >> gcmc.log 2>&1 & echo $!' %(gcmcscript, inp)
    else:
        print ("Syntax:")
        print ("launch.py -[ga/gcmc] -[ga.inp/gcmc.inp]")
        print ("e.g. launch.py -ga ga.inp")
        print ("e.g. launch.py -gcmc gcmc.inp")
        raise SyntaxError 
        
    #Get job pid
    pid = subprocess.Popen(cmd, stdout=subprocess.PIPE,  shell=True) 
    pid = pid.stdout.read().strip().decode('utf-8').split()[0]
    
    #Get hostname 
    cmd = 'hostname'
    host = subprocess.Popen(cmd, stdout=subprocess.PIPE,  shell=True) 
    host = host.stdout.read().strip().decode('utf-8').split()[0]
    
    with open('ga.pid', 'w') as f:
        print ('%s starts running at %s' %(mode, host))
        print ('The status of %s will be recorded in %s.log'  %(mode, mode))
        
        print ('To terminate the current %s simulation:' %(mode))
        print ("1. login to %s by: ssh %s" %(host, host))
        print ("2. then terminate the %s job by: kill %s" %(mode, pid))        
        
        f.write('To terminate the current %s simulation:\n' %(mode))
        f.write("1. login to %s by: ssh %s\n" %(host, host))
        f.write("2. then terminate the %s job by: kill %s\n" %(mode, pid))