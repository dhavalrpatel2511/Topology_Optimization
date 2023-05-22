"""
Author of code:     Zhi Hao Zuo and Yi Min Xie (2015)
                    "A simple and compact Python code for complex 3D topology optimization"
Modified by:        Dhaval Patel et al. (2023)
                    "Single and Multi-Material Topology Optimization of Continuum Structures: ABAQUS Plugin"
Purpose:            Single material Topology Optimization
Procedure:          i) Create the .cae file that shall contain a model 'Model-1' 
                    with a dependent part 'Part-1' and a static step 'Step-1'.
                    ii) Run this python file in Abqus cae environment, it will ask 
                    for some input parameters.
Last modified date: 22.05.2023
Modification:       preFlt function computationally effiecient by using ABAQUS inbuilt 
                    function 'getByBoundingBox'. Value of Ae (sensitivity number) is 
                    modified according to new BESO algorithm.
Repository URL:     https://github.com/dhavalrpatel2511/Topology_Optimization.git
"""
## Important libraries to intereact with ABAQUS environment
from abaqus import getInput,getInputs
from odbAccess import openOdb
import math,customKernel
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from math import *
import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
import os
import sys

## Function of formatting Abaqus model for stiffness optimization
def fmtMdb(Mdb):
  mdl = Mdb.models['Model-1']
  part = mdl.parts['Part-1']
  # Build sections and assign solid section
  mdl.Material('Material01').Elastic(((1000, 0.3), ))
  mdl.HomogeneousSolidSection('sldSec','Material01')
  mdl.Material('Material02').Elastic(((0.000000001, 0.3), ))
  mdl.HomogeneousSolidSection('voidSec','Material02')
  part.SectionAssignment(part.Set('ss',part.elements),'sldSec')
  # Define output request
  mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
  mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))

## Function of running FEA for raw sensitivities and objective function
def FEA(Iter,Mdb,Xe,Ae,BCE_LABELS):
  Mdb.Job('Design_Job'+str(Iter),'Model-1').submit()
  Mdb.jobs['Design_Job'+str(Iter)].waitForCompletion()
  opdb = openOdb('Design_Job'+str(Iter)+'.odb')
  seng = opdb.steps['Step-1'].frames[-1].fieldOutputs['ESEDEN'].values
  for en in seng: Ae[en.elementLabel]=en.data*0.5
  obj=opdb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]
  opdb.close()
  return obj

## Function to find the neighbouring elments of each elment and thier weightage
def preFlt(Rmin,Elmts,Nds,Fm):
  import numpy as np
  # Calculate element centre coordinates
  elm, c0 = np.zeros(len(Elmts)), np.zeros((len(Elmts),3))
  CR = Rmin + 1.0
  for i in range(len(elm)):
    elm[i] = Elmts[i].label
    nds = Elmts[i].connectivity
    for nd in nds: c0[i] = np.add(c0[i],np.divide(Nds[nd].coordinates,len(nds)))
  # Weighting factors
  for i in range(len(elm)):
    xmin,xmax,ymin,ymax,zmin,zmax = 0.0,0.0,0.0,0.0,0.0,0.0
    elm_list = []
    xmin = c0[i][0] - CR
    xmax = c0[i][0] + CR
    ymin = c0[i][1] - CR
    ymax = c0[i][1] + CR
    zmin = c0[i][2] - CR
    zmax = c0[i][2] + CR
    elements_list = part.elements.getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
    for elep in elements_list:
        elm_list.append(elep.label) 
    Fm[elm[i]] = [[],[]]
    for j in elm_list:
       dis = np.sqrt(np.sum(np.power(np.subtract(c0[i],c0[j-1]),2)))
       if dis<Rmin:
         Fm[elm[i]][0].append(j)
         Fm[elm[i]][1].append(Rmin - dis)
    Fm[elm[i]][1] = np.divide(Fm[elm[i]][1],np.sum(Fm[elm[i]][1]))

## Function of filtering sensitivities
def fltAe(Ae,Fm):
  raw = Ae.copy()
  for el in Fm.keys():
    Ae[el] = 0.0
    for i in range(len(Fm[el][0])): Ae[el]+=raw[Fm[el][0][i]]*Fm[el][1][i]

## Function of optimality update for design variables and Abaqus model
def BESO(Vf,Xe,Ae,Part,Elmts,BCE_LABELS):
  lo, hi = min(Ae.values()), max(Ae.values())
  tv = Vf*len(Elmts)
  while (hi-lo)/hi > 1.0e-5:
    th = (lo+hi)/2.0
    for key in Xe.keys(): 
        if key in BCE_LABELS:
           Xe[key] = 1.0
        else:   
           Xe[key] = 1.0 if Ae[key]>th else 0.001
    if sum(Xe.values())-tv>0: lo = th
    else: hi = th
  # Label elements as solid or void
  vlb, slb = [], []
  for el in Elmts:
    if Xe[el.label] == 1.0: slb.append(el.label)
    else: vlb.append(el.label)
  # Assign solid and void elements to each section
  Part.SectionAssignment(Part.SetFromElementLabels('ss',slb),'sldSec')
  Part.SectionAssignment(Part.SetFromElementLabels('vs',vlb),'voidSec')

## ====== MAIN PROGRAM ======
if __name__ == '__main__':
  # Set parameters and inputs
  pars = (('VolFrac:','0.4'), ('Rmin:', '3'), ('ER:', '0.02'))
  vf,rmin,ert = [float(k) if k!=None else 0 for k in getInputs(pars,dialogTitle='Parameters')]
  if vf<=0 or rmin<0 or ert<=0: sys.exit()
  mddb = openMdb(getInput('Input CAE file:',default='hypermesh_surfed.cae'))
  # Design initialization
  fmtMdb(mddb)
  part = mddb.models['Model-1'].parts['Part-1']
  elmts, nds = part.elements, part.nodes
  oh, vh = [], []
  xe, ae, oae, fm = {}, {}, {}, {}
  for el in elmts: xe[el.label] = 1.0
  if rmin>0: preFlt(rmin,elmts,nds,fm)
  print('hello')
  # Optimization iteration
  change, iter, obj = 1, -1, 0
  BCE_LABELS=[]
  while change > 0.0001 and iter<=100:
    iter += 1
    # Run FEA
    oh.append(FEA(iter,mddb,xe,ae,BCE_LABELS))
    # Process sensitivities
    if rmin>0: fltAe(ae,fm)
    if iter > 0: ae=dict([(k,(ae[k]+oae[k])/2.0) for k in ae.keys()])
    oae = ae.copy()
    # BESO optimization
    vh.append(sum(xe.values())/len(xe))
    nv = max(vf,vh[-1]*(1.0-ert))
    BESO(nv,xe,ae,part,elmts,BCE_LABELS)
    #if nv==vf: change=oh[-1]-oh[iter]
    if nv==vf: change=math.fabs((sum(oh[iter-4:iter+1])-sum(oh[iter-9:iter-4]))/sum(oh[iter-9:iter-4]))

# Save results
mddb.customData.History = {'vol':vh,'obj':oh}
mddb.saveAs('Final_design.cae')