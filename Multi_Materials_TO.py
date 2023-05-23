"""
Author of code:     Zhi Hao Zuo and Yi Min Xie (2015)
                    "A simple and compact Python code for complex 3D topology optimization"
Modified by:        Dhaval Patel et al. (2023)
                    "Single and Multi-Material Topology Optimization of Continuum Structures: ABAQUS Plugin"
Purpose:            Multi materials Topology Optimization
Procedure:          i) Create the .cae file that shall contain a model 'Model-1' 
                    with a dependent part 'Part-1' and a static step 'Step-1'.
                    ii) Run this python file in Abqus cae environment, it will ask 
                    for some input parameters.
Last modified date: 22.05.2023
Modification:       fmtMdb function computationally effiecient by using ABAQUS inbuilt 
                    function 'getByBoundingBox'. Value of Ae (sensitivity number) is 
                    modified according to new BESO algorithm.
Repository URL:     https://github.com/dhavalrpatel2511/Topology_Optimization.git
"""
## Important libraries to intereact with ABAQUS environment
import math,customKernel
import numpy as np
from abaqus import getInput,getInputs
from odbAccess import openOdb

### Function of formatting Abaqus Model 
def fmtMdb(mdl,part,Elmts,Nds,Y1,Y2,Y3,PR,Rmin,Fm):
  # Build sections and assign solid section
  mdl.Material('Material01').Elastic(((Y1, PR), ))
  mdl.HomogeneousSolidSection('sldSec','Material01')
  mdl.Material('Material02').Elastic(((Y2, PR), ))
  mdl.HomogeneousSolidSection('softSec','Material02')
  mdl.Material('Material03').Elastic(((Y3, PR), ))
  mdl.HomogeneousSolidSection('voidSec','Material03')
  part.SectionAssignment(part.Set('ss',part.elements),'sldSec')
  # Define output request
  mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
  mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))
  # Calculate element centre coordinates
  elm, c0 = np.zeros(len(Elmts)), np.zeros((len(Elmts),3))
  CR = Rmin + 2.0
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

### Function of running FEA to find raw sensitivities and objective function
def FEA(Iter,Mdb,Ae,Ae1,RY21,RY31,Fm):
  Mdb.Job('Design_Job'+str(Iter),'Model-1').submit()
  Mdb.jobs['Design_Job'+str(Iter)].waitForCompletion()
  opdb = openOdb('Design_Job'+str(Iter)+'.odb')
  seng = opdb.steps['Step-1'].frames[-1].fieldOutputs['ESEDEN'].values
  for en in seng:
      Ae[en.elementLabel]=en.data*(1-RY21)/0.5
      Ae1[en.elementLabel]=en.data*(1-RY31)/0.5
  obj=opdb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]
  opdb.close()
  ## Function of filtering sensitivities
  raw = Ae.copy()
  raw1 = Ae1.copy()
  for el in Fm.keys():
    Ae[el] = 0.0
    Ae1[el] = 0.0
    for i in range(len(Fm[el][0])):
        Ae[el]+=raw[Fm[el][0][i]]*Fm[el][1][i]
        Ae1[el]+=raw1[Fm[el][0][i]]*Fm[el][1][i]
  return obj

## Function of optimality update for design variables and Abaqus model
def BESO(Vf,Xe,Ae1,Part,Elmts,Case):
  lo, hi = min(Ae1.values()), max(Ae1.values())
  tv = Vf*len(Elmts)
  while (hi-lo)/hi > 1.0e-5:
    th = (lo+hi)/2.0
    for key in Xe.keys(): Xe[key] = 1.0 if Ae1[key]>th else 0.00001
    if sum(Xe.values())-tv>0: lo = th
    else: hi = th
  # Label elements as solid or void
  vlb, slb = [], []
  for el in Elmts:
    if Xe[el.label] == 1.0: slb.append(el.label)
    else: vlb.append(el.label)
  # Assign solid and void elements to each section
  Part.SectionAssignment(Part.SetFromElementLabels('ss',slb),'sldSec')
  if Case == 1 or Case == 3:
     Part.SectionAssignment(Part.SetFromElementLabels('vs',vlb),'voidSec')
  else:    
     Part.SectionAssignment(Part.SetFromElementLabels('sos',vlb),'softSec')

## Function of optimality update for design variables and Abaqus model
def BESO1(Vf,Vf1,Xe,nXe,Ae,Ae1,nAe,Part,Elmts):
  lo, hi = min(Ae1.values()), max(Ae1.values())
  tv = Vf*len(Elmts)
  while (hi-lo)/hi > 1.0e-5:
    th = (lo+hi)/2.0
    for key in Xe.keys(): Xe[key] = 1.0 if Ae1[key]>th else 0.00001
    if sum(Xe.values())-tv>0: lo = th
    else: hi = th
  # Label elements as solid or void
  vlb, slb, soslb = [], [], []
  for el in Elmts:
    if Xe[el.label] == 0.00001:
       vlb.append(el.label)
    else:
       nXe[el.label] = 1.0
       nAe[el.label] = Ae[el.label]
  #     
  lo1, hi1 = min(nAe.values()), max(nAe.values())
  tv1 = Vf1
  while (hi1-lo1)/hi1 > 1.0e-5:
    th1 = (lo1+hi1)/2.0
    for key in nXe.keys(): nXe[key] = 1.0 if nAe[key]>th1 else 0.00001
    if sum(nXe.values())-tv1>0: lo1 = th1
    else: hi1 = th1
  #  
  for key in nXe.keys():
    if nXe[key] == 1.0:
       slb.append(key)
    else:
       soslb.append(key)
  # Assign solid and void elements to each section
  Part.SectionAssignment(Part.SetFromElementLabels('ss',slb),'sldSec')
  Part.SectionAssignment(Part.SetFromElementLabels('sos',soslb),'softSec')
  Part.SectionAssignment(Part.SetFromElementLabels('vs',vlb),'voidSec')

## ====== MAIN PROGRAM ======
if __name__ == '__main__':
  # Set parameters and inputs
  pars = (('VF_Stiff:','0.40'), ('VF_Soft:','0.40'), ('Rmin:', '3'), ('ER:', '0.02'), ('YM_Stiff:', '1000'), ('YM_Soft:', '100'), ('YM_Void:', '0.000001'), ('PoissonsRatio:', '0.3'), ('Case:', '3'))
  vf,vf1,rmin,ert,Y1,Y2,Y3,PR,Case = [float(k) if k!=None else 0 for k in getInputs(pars,dialogTitle='Parameters')]
  if vf<=0 or rmin<0 or ert<=0 or Y1<=0 or Y2<=0 or Y3<=0 or PR<=0: sys.exit()
  mddb = openMdb(getInput('Input CAE file:',default='Model_2D.cae'))
  mdl = mddb.models['Model-1']
  part = mdl.parts['Part-1']
  Elmts, Nds = part.elements, part.nodes
  # Design initialization
  RY21 = Y2/Y1
  RY31 = Y3/Y1
  nae, oae, oae1, fm = {}, {}, {}, {}
  fmtMdb(mdl,part,Elmts, Nds,Y1,Y2,Y3,PR,rmin,fm)
  oh, vh, vh1 = [], [], []
  total = len(Elmts)*0.40
  vh1.append(total)
  xe, nxe, ae, ae1 = {}, {}, {}, {}
  for el in Elmts: xe[el.label] = 1.0
  # Optimization iteration
  change, iter, obj, ccase, ccase1 = 1, -1, 0, 0, 0
  while change > 0.0001:
    iter += 1
    # Run FEA
    oh.append(FEA(iter,mddb,ae,ae1,RY21,RY31,fm))
    # Process sensitivities
    if iter > 0:
       ae=dict([(k,(ae[k]+oae[k])/2.0) for k in ae.keys()])
       ae1=dict([(k,(ae1[k]+oae1[k])/2.0) for k in ae1.keys()])
    oae = ae.copy()
    oae1 = ae1.copy()
    # BESO optimization
    vh.append(sum(xe.values())/len(xe))
    if (vh[-1]*(1.0-ert)) <= vf:
       ccase1 = ccase1 + 1
    nv = max(vf,vh[-1]*(1.0-ert))
    if nv == vf and ccase == 0:
       vf = vf1
       ccase = ccase + 1
    if ccase == 0 or ccase1 == 1 or Case == 1 or Case == 2:
       if Case==2:
          BESO(nv,xe,ae,part,Elmts,Case)
       else:
          BESO(nv,xe,ae1,part,Elmts,Case)          
       ccase1 = ccase1 + 1
    else:
       vh1.append(vh1[-1]*(0.99))
       nv1 = max(720,vh1[-1])
       nxe={}
       nae={}
       BESO1(nv,nv1,xe,nxe,ae,ae1,nae,part,Elmts)
    #if nv==vf: change=oh[-1]-oh[iter]
    if iter>10: change=math.fabs((sum(oh[iter-4:iter+1])-sum(oh[iter-9:iter-4]))/sum(oh[iter-9:iter-4]))

# Save results
mddb.customData.History = {'vol':vh,'obj':oh}
mddb.saveAs('Final_design.cae')