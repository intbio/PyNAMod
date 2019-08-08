#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Простой пример использования py3dna
'''
from py3dna import py3dna
# Класс написан так, что может работать из VMD
# и всемсто PDB принимать atomsel (VMD_atomsel=atomsel), может быть удобно
# для расчета энергии деформации ДНК по трактории
# Инициализируем  объект из PDB
dna=py3dna('h1_seq.pdb',tempdir='/usertemp',path='/home/armeev/Software/Source/x3dna-v2.1')
    
# или в VMD
from py3dna import py3dna
from VMD import *
from Molecule import *
from atomsel import *
from animate import *
dnaSel=atomsel('nucleic')
dna=py3dna(VMD_atomsel=dnaSel,tempdir='temp/')
#    ,path='/home/armeev/Software/Source/x3dna-v2.1')

# Включаем участки ДНК в процедуру минимизации
# Здесь выбираем участки с 1 по 40 и 107 по 147 нп включительно
# шаги между ними будут изменяться при минимизации
dna.set_movable_bp([[1,50],[177,226]])

#Выбираем расстояния, которые будут ограничиваться при минимизации
# Здесь мы ограничиваем расстояние между 1 и 147 нп 80 А
# и расстояние между 10 и 137 нп 75 А
dna.set_pairs_list_and_dist([[21,211],[23,205],[26,203],[31,200]],[50,52,47,47])

#Минимизируем конформацию без учета расстояний
frame,result=dna.run_minimize(frame,usepairs=False,method='BFGS',options={'maxiter':1000,'maxfev':1000,'eps':.1,'xtol' :1.0, 'disp': True})
options={'maxiter':100,'maxfev': 100,'xtol' :1.0, 'disp': True}
dna.distCoef=1.0
# пишем конформацию в PDB
# путь должен быть абсолютным, или файл будет записан в TEMP
dna.frame_to_pdb(frame,'/home/armeev/Software/Source/py3dna/test/CEN3SEQ_MINIM.PDB')

#Продолжаем минимизацию, включив ограничнеия по парам
#Ограничив количество итераций алгоритма минимизации
frame,result=dna.run_minimize(frame,usepairs=True,use_dyes=True,vmdload=False,method='BFGS',options={'maxfev':30,'eps':0.1, 'disp': True})

options={'maxiter':20,'maxfev': 20,'xtol' :2.0, 'disp': True},method='Powell'
dna.frame_to_pdb(frame,'/home/armeev/Software/Source/py3dna/test/result_dist2.pdb')
nucleic and resid 0 1 and chain I J or  
nucleic and resid 74 75 221 222 and chain A B
