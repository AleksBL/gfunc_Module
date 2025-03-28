#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:54:53 2021

@author: aleks
"""

import numpy as np
import matplotlib.pyplot as plt
from Block_matrices.Block_matrices import block_td,block_sparse,test_partition_2d_sparse_matrix, slices_to_npslices
from Block_matrices.Block_matrices import Build_BTD_vectorised, Build_BS_vectorised, Blocksparse2Numpy, sparse_find_faster
from Block_matrices.Block_matrices import Build_BTD_purenp, Build_BS_purenp
import scipy.sparse as sp
import sisl
from time import time
from tqdm import tqdm
import itertools
# from siesta_python.funcs import find_np_list
def find_np_list(L, val):
    return [i for i, x in enumerate(L) if (x == val).all()]

def CZ(s,dt = complex):
    return np.zeros(s, dtype = dt)

class Greens_function:
    def __init__(self, 
                 sisl_H,
                 pivot, 
                 P,
                 sisl_S = None):
        
        
        self.H = sisl_H
        self.S = sisl_S
        self.pivot = pivot
        self.P = P
        
        # self.dropped_indices = drop_indices
        # the indices to keep in the end. We might want to throw 
        # away some indecies (usually the electrode/buffer orbitals in H ):
        # self.kept_indices = [[i for i in range(sisl_H.no) if not i in drop_indices[0]],
                             # [i for i in range(sisl_H.no) if not i in drop_indices[1]]]
        
    def set_eta(self, eta):
        self.eta = eta
    
    def set_ev(self,Eg):
        self.E = Eg + 1j * self.eta
        if len(Eg) != self.SE[0][0].shape[1]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
        self.ne = len(self.E)
     
    def set_kv(self,k_in):
        self.k_avg = k_in.copy()
        self.nk    = len(self.k_avg)
        if self.nk==1 and isinstance(k_in,list) and k_in==[None]:
            print('\n No k-points!\n')
            self.kvecs  = None
        else:
            self.kvecs = k_in.copy()
        if len(k_in) != self.SE[0][0].shape[0]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
    
    def set_SE(self, SE, SE_inds):
        self.SE = SE
        self.SE_inds = SE_inds
        self.n_elecs = len(self.SE[0])
        if len(SE)>1:
            assert len(self.SE[0]) == len(self.SE[1])
        
        # make sure the SE s are sorted like your E and k grids.
    
    
    def iG(self, spin, test_interval = 100, tol = 1e-12, force_continue = False, 
           duplicate_test = False, dtype = np.complex128, 
           #eidx = None,
           #kidx = None
           safe = True,
           nolowdin=False):
        
        piv  = self.pivot[spin]
        npiv = len(piv)
        P    = np.array(self.P[spin],dtype = np.int32)
        
        n_diags = len(P)-1
        kidx = None
        eidx = None
        
        if kidx is None:
            nk = self.nk
            kidx = [i for i in range(nk)]
        else:
            nk = len(kidx)
        
        if eidx is None:
            ne = self.ne
            energy_idx=[i for i in range(ne)]
        else:
            ne = len(eidx)
            energy_idx = [i for i in eidx]
        
        n_elecs = self.n_elecs
        H  = self.H
        Ov  = self.S
        no = H.no
        # The BTD blocks gets initialised:
        Ia = [i for i in range(n_diags  )]
        Ib = [i for i in range(n_diags-1)]
        Ic = [i for i in range(n_diags-1)]
        
        Al   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
        Bl   =  [CZ((nk,ne,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
        Cl   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
        
        print('\n Building ES - H - Self Energies \n')
        iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False, E_grid = self.E)
        
        del Al, Bl, Cl
        
        if True:
            L1_inds = []; L2_inds = []
            L1_vals = []; L2_vals = []
            for _i in range(n_diags):
                for _j in range(n_diags):
                    L1_inds += [(_i,_j)]
                    L1_vals += [np.zeros((nk, 1, P[_i+1]-P[_i  ],P[_j+1]-P[ _j ]), dtype = np.complex128)]
                    L2_inds += [(_i,_j)]
                    L2_vals += [np.zeros((nk, 1, P[_i+1]-P[_i  ],P[_j+1]-P[ _j ]), dtype = np.complex128)]
            
            Al   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
            Bl   =  [CZ((nk,1,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
            Cl   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
            Hamiltonian = block_td(Al,Bl,Cl,Ia.copy(),Ib.copy(),Ic.copy(),diagonal_zeros=False, E_grid = self.E)
            
            Als   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
            Bls   =  [CZ((nk,1,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
            Cls   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
            BTD_Overlap = block_td(Als,Bls,Cls,Ia.copy(),Ib.copy(),Ic.copy(),diagonal_zeros=False, E_grid = self.E)
            del Al, Bl, Cl, Als, Bls, Cls
        
        I            = sp.identity(no).tocsr()
        idx_coupling = []
        f, S = test_partition_2d_sparse_matrix(sp.identity(npiv).tocsr(),P)
        nS   = slices_to_npslices(S)
        
        for e in range(n_elecs):
            idx_e = self.SE_inds[spin][e]
            iv, jv = [], []
            for ii in idx_e:
                for jj in idx_e:
                    iv+=[ii]
                    jv+=[jj]
            idx_coupling += [[np.array(iv),np.array(jv)]]
        # Needed in the end of the i loop
        SM_val_list = []
        SM_ind_list = []
        
        SE_val_list = []
        SE_ind_list = []
        print(kidx)
        for i in kidx:
            k = self.kvecs[i]
            if 'unpolarized' in H.spin.__str__():
                hk  =  H.Hk( k = k)
            else:
                hk  =  H.Hk(spin = spin, k = k)
            if Ov is not None:
                sk = Ov.Sk(k = k)
                print('Using S = S')
                if i == 0:
                    print('\n Overlap Included!\n')
            else:
                print('Using S = I')
                sk = I
                
            
            if True:
                i1, j1, d1 = [],[],[]
                i2, j2, d2 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                i3, j3, d3 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                eps = sp.csr_matrix((no,no),dtype=dtype)
                if safe:
                    for j in energy_idx:
                        z = self.E[j]
                        se_list    = []
                        gamma_list = []
                        for e in range(n_elecs):
                            se_sub = self.SE[spin][e][i, j, :, :]
                            se_sub[np.abs(se_sub)<1e-15] = (0.1 + 0.1j) * 1e-15
                            iv,jv = idx_coupling[e]
                            se_sparse  = sp.csr_matrix((se_sub.ravel(), (iv,jv)), shape = (no,no),dtype = dtype)
                            se_list   += [  se_sparse.copy()  ]
                            bm_ele = se_sub - se_sub.conj().T
                            bm_ele[np.abs(bm_ele)<1e-15] = (0.1 + 0.1j) * 1e-15
                            gamma_list+=[1j * sp.csr_matrix((bm_ele.ravel(), (iv,jv)),shape = (no,no),dtype = dtype)]
                        iG = sk * z - hk - sum(se_list)
                        # Add eps to keep non-zero pattern consistent for stacking later
                        eps[abs(iG)>0]  += 1e-17
                        eps[eps>1.1e-17] = 1e-17
                    del iG, z, se_list, gamma_list, se_sparse, se_sub, iv, jv, bm_ele
                
                for j in energy_idx:
                    z = self.E[j]
                    se_list    = []
                    gamma_list = []
                    for e in range(n_elecs):
                        se_sub = self.SE[spin][e][i, j, :, :]
                        se_sub[np.abs(se_sub)<1e-15] = (0.1 + 0.1j) * 1e-15
                        iv,jv = idx_coupling[e]
                        se_sparse  = sp.csr_matrix((se_sub.ravel(), (iv,jv)), shape = (no,no),dtype = dtype)
                        se_list   += [  se_sparse.copy()  ]
                        
                        bm_ele = se_sub - se_sub.conj().T
                        
                        bm_ele[np.abs(bm_ele)<1e-15] = (0.1 + 0.1j) * 1e-15
                        gamma_list+=[1j * sp.csr_matrix((bm_ele.ravel(), (iv,jv)),shape = (no,no),dtype = dtype)]
                    
                    iG = sk * z - hk - sum(se_list)
                    # Add eps to keep non-zero pattern consistent for stacking later
                    
                    iG += eps
                    iG  = iG[piv, :][:, piv]
                    
                    di, dj, dv = sparse_find_faster(iG) # sp.find(iG) #, but without sheenanigans
                    
                    i1.append(di); j1.append(dj); d1.append(dv)
                    if True:#not only_greens:
                        for e in range(n_elecs):
                            #Broadening matrix
                            di,dj,dv = sp.find(gamma_list[e][piv, :][:, piv])
                            dv[np.abs(dv)<1e-15] = 0.0
                            
                            i2[e].append(di); j2[e].append(dj); d2[e].append(dv)
                            #Self energy
                            di2,dj2,dv2 = sp.find(se_list[e][piv, :][:, piv])
                            i3[e].append(di2); j3[e].append(dj2); d3[e].append(dv2)
            # We put the matrix elements into the block_td class:
            
            Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)
            
            for b in range(n_diags):
                iGreens.Al[b][i, :, :, :] += Av[b]
                if b<n_diags-1:
                    iGreens.Bl[b][i, :, :, :] += Bv[b]
                    iGreens.Cl[b][i, :, :, :] += Cv[b]
            
            # We make the block_sparse matrices for the scattering matrices:
            for e in range(n_elecs):
                
                di,dj,dv = np.vstack(i2[e]), np.vstack(j2[e]), np.vstack(d2[e])
                dv[np.abs(dv)<1e-15] = 0.0
                
                I,V = Build_BS_vectorised(di, dj, dv, P)
                
                dise,djse,dvse = np.vstack(i3[e]), np.vstack(j3[e]), np.vstack(d3[e])
                Ise,Vse = Build_BS_vectorised(dise, djse, dvse, P)
                
                if i == 0:
                    SM_val_list += [[np.zeros((nk, ) + vvv.shape, dtype = dtype) for vvv in V]]
                    SM_ind_list += [I]
                    
                    SE_val_list += [[np.zeros((nk, ) + vvv.shape, dtype = dtype) for vvv in Vse]]
                    SE_ind_list += [Ise]
                    
                    for ivvv,vvv in enumerate(V):
                        SM_val_list[e][ivvv][i,:,:,:] += vvv
                    for ivvv, vvv in enumerate(Vse):
                        SE_val_list[e][ivvv][i,:,:,:] += vvv
                    
                else:
                    for block in range(len(I)):
                        ind_block = I[block]
                        val_block = V[block]
                        where = find_np_list(I, ind_block)
                        if len(where) == 1:
                            SM_val_list[e][where[0]][i,:,:,:] += val_block
                        elif len(where) == 0:
                            SM_val_list[e].append(np.zeros((nk, ) + val_block.shape, dtype = dtype))
                            SM_ind_list[e].append(ind_block)
                            SM_val_list[e][-1][i,:,:,:] += val_block
                        else:
                            print('\n Multiply counted block: Error, you messed up\n')
                            assert 1 == 0
                    for block in range(len(Ise)):
                        ind_block = Ise[block]
                        val_block = Vse[block]
                        where = find_np_list(Ise, ind_block)
                        if len(where) == 1:
                            SE_val_list[e][where[0]][i,:,:,:] += val_block
                        elif len(where) == 0:
                            SE_val_list[e].append(np.zeros((nk, ) + val_block.shape, dtype = dtype))
                            SE_ind_list[e].append(ind_block)
                            SE_val_list[e][-1][i,:,:,:] += val_block
                        else:
                            print('\n Multiply counted block: Error, you messed up\n')
                            assert 1 == 0
            if nolowdin == False:
                esk, vsk = np.linalg.eigh(sk[piv, :][:, piv].toarray())
                s1_12   = vsk.dot(np.diag(1/np.sqrt(esk))).dot(vsk.T.conj())
                s1_12   = sp.csr_matrix(s1_12)
                s2_12   = vsk.dot(np.diag(  np.sqrt(esk))).dot(vsk.T.conj())
                s2_12   = sp.csr_matrix(s2_12)
                S1_i, S1_j, S1_v = sparse_find_faster(s1_12)
                S2_i, S2_j, S2_v = sparse_find_faster(s2_12)
                I_S1,V_S1 = Build_BS_purenp(S1_i, S1_j, S1_v, P)
                I_S2,V_S2 = Build_BS_purenp(S2_i, S2_j, S2_v, P)
                
                for count in range(len(I_S1)):
                    ind, val = I_S1[count], V_S1[count]
                    ind = (ind[0], ind[1])
                    L1_vals[L1_inds.index(ind)][i,0,:,:] += val
                for count in range(len(I_S2)):
                    ind, val = I_S2[count], V_S2[count]
                    ind = (ind[0], ind[1])
                    L2_vals[L2_inds.index(ind)][i,0,:,:] += val
                
                Lowdin =  [block_sparse(L1_inds, L1_vals, iGreens.Block_shape),
                           block_sparse(L2_inds, L2_vals, iGreens.Block_shape)
                          ]
                
            iH, jH, vH = sparse_find_faster(hk[piv,:][:,piv])
            iS, jS, vS = sparse_find_faster(sk[piv,:][:,piv])
            #print(sk[piv,:][:,piv][:5, :5])
            Ai, Bi, Ci = Build_BTD_purenp(iH, jH, vH,    nS)
            Ais, Bis, Cis = Build_BTD_purenp(iS, jS, vS, nS)
            
            for bb in range(n_diags):
                Hamiltonian.Al[bb][i,0,:,:] += Ai[bb]
                BTD_Overlap.Al[bb][i,0,:,:] += Ais[bb]
                if bb < n_diags -1:
                    Hamiltonian.Bl[bb][i,0,:,:] += Bi[bb]
                    Hamiltonian.Cl[bb][i,0,:,:] += Ci[bb]
                    BTD_Overlap.Bl[bb][i,0,:,:] += Bis[bb]
                    BTD_Overlap.Cl[bb][i,0,:,:] += Cis[bb]
        
        Gammas        = [block_sparse(SM_ind_list[e], SM_val_list[e], iGreens.Block_shape) for e in range(n_elecs)]
        self_energies = [block_sparse(SE_ind_list[e], SE_val_list[e], iGreens.Block_shape) for e in range(n_elecs)]
        if nolowdin==False:
            return iGreens, Gammas, Lowdin, Hamiltonian, BTD_Overlap, self_energies
        else:
            return iGreens, Gammas, Hamiltonian, BTD_Overlap, self_energies


class Greens_function_minimal:
    def __init__(self, 
                 sisl_H,
                 pivot, 
                 P,
                 sisl_S = None):
        
        
        self.H = sisl_H
        self.S = sisl_S
        self.pivot = pivot
        self.P = P
        
        # self.dropped_indices = drop_indices
        # the indices to keep in the end. We might want to throw 
        # away some indecies (usually the electrode/buffer orbitals in H ):
        # self.kept_indices = [[i for i in range(sisl_H.no) if not i in drop_indices[0]],
                             # [i for i in range(sisl_H.no) if not i in drop_indices[1]]]
    
    def set_eta(self, eta):
        self.eta = eta
    
    def set_ev(self,Eg):
        self.E = Eg + 1j * self.eta
        if len(Eg) != self.SE[0][0].shape[1]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
        self.ne = len(self.E)
     
    def set_kv(self,k_in):
        self.k_avg = k_in.copy()
        self.nk    = len(self.k_avg)
        if self.nk==1 and isinstance(k_in,list) and k_in==[None]:
            print('\n No k-points!\n')
            self.kvecs  = None
        else:
            self.kvecs = k_in.copy()
        if len(k_in) != self.SE[0][0].shape[0]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
    
    def set_SE(self, SE, SE_inds):
        self.SE = SE
        self.SE_inds = SE_inds
        self.n_elecs = len(self.SE[0])
        if len(SE)>1:
            assert len(self.SE[0]) == len(self.SE[1])
        
        # make sure the SE s are sorted like your E and k grids.
    
    
    def iG_minimal(self, spin, test_interval = 100, tol = 1e-12, force_continue = False, 
                   duplicate_test = False, dtype = np.complex128, 
                   ):
        
        piv  = self.pivot[spin]
        npiv = len(piv)
        P    = np.array(self.P[spin],dtype = np.int32)
        
        n_diags = len(P)-1
        kidx = None
        eidx = None
        
        if kidx is None:
            nk = self.nk
            kidx = [i for i in range(nk)]
        else:
            nk = len(kidx)
        
        if eidx is None:
            ne = self.ne
            energy_idx=[i for i in range(ne)]
        else:
            ne = len(eidx)
            energy_idx = [i for i in eidx]
        
        n_elecs = self.n_elecs
        H  = self.H
        Ov  = self.S
        no = H.no
        # The BTD blocks gets initialised:
        Ia = [i for i in range(n_diags  )]
        Ib = [i for i in range(n_diags-1)]
        Ic = [i for i in range(n_diags-1)]
        
        Al   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
        Bl   =  [CZ((nk,ne,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
        Cl   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
        
        print('\n Building ES - H - Self Energies \n')
        iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False, E_grid = self.E)
        
        del Al, Bl, Cl
        
        I            = sp.identity(no).tocsr()
        idx_coupling = []
        f, S = test_partition_2d_sparse_matrix(sp.identity(npiv).tocsr(),P)
        nS   = slices_to_npslices(S)
        
        for e in range(n_elecs):
            idx_e = self.SE_inds[spin][e]
            iv, jv = [], []
            for ii in idx_e:
                for jj in idx_e:
                    iv+=[ii]
                    jv+=[jj]
            idx_coupling += [[np.array(iv),np.array(jv)]]
        # Needed in the end of the i loop
        SM_val_list = []
        SM_ind_list = []
        
        SE_val_list = []
        SE_ind_list = []
        print(kidx)
        for i in kidx:
            k = self.kvecs[i]
            if 'unpolarized' in H.spin.__str__():
                hk  =  H.Hk( k = k)
            else:
                hk  =  H.Hk(spin = spin, k = k)
            if Ov is not None:
                sk = Ov.Sk(k = k)
                if i == 0:
                    print('\n Overlap Included!\n')
            else:
                sk = I
            
            if True:
                i1, j1, d1 = [],[],[]
                i2, j2, d2 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                i3, j3, d3 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                
                for j in energy_idx:
                    z = self.E[j]
                    se_list    = []
                    gamma_list = []
                    for e in range(n_elecs):
                        se_sub = self.SE[spin][e][i, j, :, :]
                        se_sub[np.abs(se_sub)<1e-15] = (0.1 + 0.1j) * 1e-15
                        iv,jv = idx_coupling[e]
                        se_sparse  = sp.csr_matrix((se_sub.ravel(), (iv,jv)), shape = (no,no),dtype = dtype)
                        se_list   += [  se_sparse.copy()  ]
                        bm_ele = se_sub - se_sub.conj().T
                        bm_ele[np.abs(bm_ele)<1e-15] = (0.1 + 0.1j) * 1e-15
                        gamma_list+=[1j * sp.csr_matrix((bm_ele.ravel(), (iv,jv)),shape = (no,no),dtype = dtype)]
                    
                    iG = sk * z - hk - sum(se_list)
                    iG = iG[piv, :][:, piv]
                    di, dj, dv = sparse_find_faster(iG) # sp.find(iG) #, but without sheenanigans
                    i1.append(di); j1.append(dj); d1.append(dv)
                    if True:#not only_greens:
                        for e in range(n_elecs):
                            #Broadening matrix
                            di,dj,dv = sp.find(gamma_list[e][piv, :][:, piv])
                            dv[np.abs(dv)<1e-15] = 0.0
                            
                            i2[e].append(di); j2[e].append(dj); d2[e].append(dv)

            # We put the matrix elements into the block_td class:
            Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)
            for b in range(n_diags):
                iGreens.Al[b][i, :, :, :] += Av[b]
                if b<n_diags-1:
                    iGreens.Bl[b][i, :, :, :] += Bv[b]
                    iGreens.Cl[b][i, :, :, :] += Cv[b]
            
            # We make the block_sparse matrices for the scattering matrices:
            for e in range(n_elecs):
                di,dj,dv = np.vstack(i2[e]), np.vstack(j2[e]), np.vstack(d2[e])
                dv[np.abs(dv)<1e-15] = 0.0
                
                I,V = Build_BS_vectorised(di, dj, dv, P)
                if i == 0:
                    SM_val_list += [[np.zeros((nk, ) + vvv.shape, dtype = dtype) for vvv in V]]
                    SM_ind_list += [I]
                    for ivvv,vvv in enumerate(V):
                        SM_val_list[e][ivvv][i,:,:,:] += vvv
                else:
                    for block in range(len(I)):
                        ind_block = I[block]
                        val_block = V[block]
                        where = find_np_list(I, ind_block)
                        if len(where) == 1:
                            SM_val_list[e][where[0]][i,:,:,:] += val_block
                        elif len(where) == 0:
                            SM_val_list[e].append(np.zeros((nk, ) + val_block.shape, dtype = dtype))
                            SM_ind_list[e].append(ind_block)
                            SM_val_list[e][-1][i,:,:,:] += val_block
                        else:
                            print('\n Multiply counted block: Error, you messed up\n')
                            assert 1 == 0
        Gammas        = [block_sparse(SM_ind_list[e], SM_val_list[e], iGreens.Block_shape) for e in range(n_elecs)]
        return iGreens, Gammas

class Greens_function_olead:
    def __init__(self, 
                 sisl_H,
                 pivot, 
                 P,
                 sisl_S = None):
        
        
        self.H = sisl_H
        self.S = sisl_S
        self.pivot = pivot
        self.P = P
        
        # self.dropped_indices = drop_indices
        # the indices to keep in the end. We might want to throw 
        # away some indecies (usually the electrode/buffer orbitals in H ):
        # self.kept_indices = [[i for i in range(sisl_H.no) if not i in drop_indices[0]],
                             # [i for i in range(sisl_H.no) if not i in drop_indices[1]]]
        
    def set_eta(self, eta):
        self.eta = eta
    
    def set_ev(self,Eg):
        self.E = Eg + 1j * self.eta
        if len(Eg) != self.SE[0][0].shape[1]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
        self.ne = len(self.E)
     
    def set_kv(self,k_in):
        self.k_avg = k_in.copy()
        self.nk    = len(self.k_avg)
        if self.nk==1 and isinstance(k_in,list) and k_in==[None]:
            print('\n No k-points!\n')
            self.kvecs  = None
        else:
            self.kvecs = k_in.copy()
        if len(k_in) != self.SE[0][0].shape[0]:
            print('Are self energies given as these indices: (idx_elec)(spin, kpnts, energy, i, j)?')
    
    def set_SE(self, SE, SE_inds):
        self.SE = SE
        self.SE_inds = SE_inds
        self.n_elecs = len(self.SE[0])
        if len(SE)>1:
            assert len(self.SE[0]) == len(self.SE[1])
        
        # make sure the SE s are sorted like your E and k grids.
    
    
    def iG(self, spin, test_interval = 100, tol = 1e-12, force_continue = False, 
           duplicate_test = False, dtype = np.complex128, 
           #eidx = None,
           #kidx = None
           safe = True,
           nolowdin=False, 
           Sig0 = None, 
           Sig1 = None,
           ):
        
        piv  = self.pivot[spin]
        npiv = len(piv)
        P    = np.array(self.P[spin],dtype = np.int32)
        
        n_diags = len(P)-1
        kidx = None
        eidx = None
        
        if kidx is None:
            nk = self.nk
            kidx = [i for i in range(nk)]
        else:
            nk = len(kidx)
        
        if eidx is None:
            ne = self.ne
            energy_idx=[i for i in range(ne)]
        else:
            ne = len(eidx)
            energy_idx = [i for i in eidx]
        
        n_elecs = self.n_elecs
        H  = self.H
        Ov = self.S
        no = H.no
        # The BTD blocks gets initialised:
        Ia = [i for i in range(n_diags  )]
        Ib = [i for i in range(n_diags-1)]
        Ic = [i for i in range(n_diags-1)]
        
        Al   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
        Bl   =  [CZ((nk,ne,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
        Cl   =  [CZ((nk,ne,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
        
        print('\n Building ES - H - Self Energies \n')
        iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False, E_grid = self.E)
        
        del Al, Bl, Cl
        
        if True:
            L1_inds = []; L2_inds = []
            L1_vals = []; L2_vals = []
            for _i in range(n_diags):
                for _j in range(n_diags):
                    L1_inds += [(_i,_j)]
                    L1_vals += [np.zeros((nk, 1, P[_i+1]-P[_i  ],P[_j+1]-P[ _j ]), dtype = np.complex128)]
                    L2_inds += [(_i,_j)]
                    L2_vals += [np.zeros((nk, 1, P[_i+1]-P[_i  ],P[_j+1]-P[ _j ]), dtype = np.complex128)]
            
            Al   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
            Bl   =  [CZ((nk,1,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
            Cl   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
            Hamiltonian = block_td(Al,Bl,Cl,Ia.copy(),Ib.copy(),Ic.copy(),diagonal_zeros=False, E_grid = self.E)
            
            Als   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags  )]
            Bls   =  [CZ((nk,1,P[i+2]-P[i+1],P[i+1]-P[i  ]), dt = dtype) for i in range(n_diags-1)]
            Cls   =  [CZ((nk,1,P[i+1]-P[i  ],P[i+2]-P[i+1]), dt = dtype) for i in range(n_diags-1)]
            BTD_Overlap = block_td(Als,Bls,Cls,Ia.copy(),Ib.copy(),Ic.copy(),diagonal_zeros=False, E_grid = self.E)
            del Al, Bl, Cl, Als, Bls, Cls
        
        I            = sp.identity(no).tocsr()
        idx_coupling = []
        f, S = test_partition_2d_sparse_matrix(sp.identity(npiv).tocsr(),P)
        nS   = slices_to_npslices(S)
        
        for e in range(n_elecs):
            idx_e = self.SE_inds[spin][e]
            iv, jv = [], []
            for ii in idx_e:
                for jj in idx_e:
                    iv+=[ii]
                    jv+=[jj]
            idx_coupling += [[np.array(iv),np.array(jv)]]
        # Needed in the end of the i loop
        SM_val_list = []
        SM_ind_list = []
        
        SE_val_list = []
        SE_ind_list = []
        print(kidx)
        for i in kidx:
            k = self.kvecs[i]
            if 'unpolarized' in H.spin.__str__():
                hk  =  H.Hk( k = k)
            else:
                hk  =  H.Hk(spin = spin, k = k)
            if Ov is not None:
                sk = Ov.Sk(k = k)
                print('Using S = S')
                if i == 0:
                    print('\n Overlap Included!\n')
            else:
                print('Using S = I')
                sk = I
            hk = hk.astype(np.complex128)
            sk = sk.astype(np.complex128)
            
            if Sig0 is not None and Sig1 is not None:
                for ielec in range(n_elecs):
                    iv,jv       = idx_coupling[ielec]
                    hk[iv, jv] += Sig0[spin][ielec][i].ravel()
                    sk[iv, jv] -= Sig1[spin][ielec][i].ravel()
                
            if True:
                i1, j1, d1 = [],[],[]
                i2, j2, d2 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                i3, j3, d3 = [[] for e in range(n_elecs)],[[] for e in range(n_elecs)],[[] for e in range(n_elecs)]
                eps = sp.csr_matrix((no,no),dtype=dtype)
                if safe:
                    for j in energy_idx:
                        z = self.E[j]
                        se_list    = []
                        gamma_list = []
                        for e in range(n_elecs):
                            se_sub = self.SE[spin][e][i, j, :, :]
                            se_sub[np.abs(se_sub)<1e-15] = (0.1 + 0.1j) * 1e-15
                            if Sig0 is not None and Sig1 is not None:
                                se_sub -= Sig0[spin][e][i] + Sig1[spin][e][i]*z
                            
                            iv,jv = idx_coupling[e]
                            se_sparse  = sp.csr_matrix((se_sub.ravel(), (iv,jv)), shape = (no,no),dtype = dtype)
                            se_list   += [  se_sparse.copy()  ]
                            bm_ele = se_sub - se_sub.conj().T
                            bm_ele[np.abs(bm_ele)<1e-15] = (0.1 + 0.1j) * 1e-15
                            gamma_list+=[1j * sp.csr_matrix((bm_ele.ravel(), (iv,jv)),shape = (no,no),dtype = dtype)]
                        iG = sk * z - hk - sum(se_list)
                        # Add eps to keep non-zero pattern consistent for stacking later
                        eps[abs(iG)>0]  += 1e-17
                        eps[eps>1.1e-17] = 1e-17
                    del iG, z, se_list, gamma_list, se_sparse, se_sub, iv, jv, bm_ele
                
                for j in energy_idx:
                    z = self.E[j]
                    se_list    = []
                    gamma_list = []
                    for e in range(n_elecs):
                        se_sub = self.SE[spin][e][i, j, :, :]
                        se_sub[np.abs(se_sub)<1e-15] = (0.1 + 0.1j) * 1e-15
                        iv,jv = idx_coupling[e]
                        se_sparse  = sp.csr_matrix((se_sub.ravel(), (iv,jv)), shape = (no,no),dtype = dtype)
                        se_list   += [  se_sparse.copy()  ]
                        
                        bm_ele = se_sub - se_sub.conj().T
                        
                        bm_ele[np.abs(bm_ele)<1e-15] = (0.1 + 0.1j) * 1e-15
                        gamma_list+=[1j * sp.csr_matrix((bm_ele.ravel(), (iv,jv)),shape = (no,no),dtype = dtype)]
                    
                    iG = sk * z - hk - sum(se_list)
                    # Add eps to keep non-zero pattern consistent for stacking later
                    
                    iG += eps
                    iG  = iG[piv, :][:, piv]
                    
                    di, dj, dv = sparse_find_faster(iG) # sp.find(iG) #, but without sheenanigans
                    
                    i1.append(di); j1.append(dj); d1.append(dv)
                    if True:#not only_greens:
                        for e in range(n_elecs):
                            #Broadening matrix
                            di,dj,dv = sp.find(gamma_list[e][piv, :][:, piv])
                            dv[np.abs(dv)<1e-15] = 0.0
                            
                            i2[e].append(di); j2[e].append(dj); d2[e].append(dv)
                            #Self energy
                            di2,dj2,dv2 = sp.find(se_list[e][piv, :][:, piv])
                            i3[e].append(di2); j3[e].append(dj2); d3[e].append(dv2)
            # We put the matrix elements into the block_td class:
            
            Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)
            
            for b in range(n_diags):
                iGreens.Al[b][i, :, :, :] += Av[b]
                if b<n_diags-1:
                    iGreens.Bl[b][i, :, :, :] += Bv[b]
                    iGreens.Cl[b][i, :, :, :] += Cv[b]
            
            # We make the block_sparse matrices for the scattering matrices:
            for e in range(n_elecs):
                
                di,dj,dv = np.vstack(i2[e]), np.vstack(j2[e]), np.vstack(d2[e])
                dv[np.abs(dv)<1e-15] = 0.0
                
                I,V = Build_BS_vectorised(di, dj, dv, P)
                
                dise,djse,dvse = np.vstack(i3[e]), np.vstack(j3[e]), np.vstack(d3[e])
                Ise,Vse = Build_BS_vectorised(dise, djse, dvse, P)
                
                if i == 0:
                    SM_val_list += [[np.zeros((nk, ) + vvv.shape, dtype = dtype) for vvv in V]]
                    SM_ind_list += [I]
                    
                    SE_val_list += [[np.zeros((nk, ) + vvv.shape, dtype = dtype) for vvv in Vse]]
                    SE_ind_list += [Ise]
                    
                    for ivvv,vvv in enumerate(V):
                        SM_val_list[e][ivvv][i,:,:,:] += vvv
                    for ivvv, vvv in enumerate(Vse):
                        SE_val_list[e][ivvv][i,:,:,:] += vvv
                    
                else:
                    for block in range(len(I)):
                        ind_block = I[block]
                        val_block = V[block]
                        where = find_np_list(I, ind_block)
                        if len(where) == 1:
                            SM_val_list[e][where[0]][i,:,:,:] += val_block
                        elif len(where) == 0:
                            SM_val_list[e].append(np.zeros((nk, ) + val_block.shape, dtype = dtype))
                            SM_ind_list[e].append(ind_block)
                            SM_val_list[e][-1][i,:,:,:] += val_block
                        else:
                            print('\n Multiply counted block: Error, you messed up\n')
                            assert 1 == 0
                    for block in range(len(Ise)):
                        ind_block = Ise[block]
                        val_block = Vse[block]
                        where = find_np_list(Ise, ind_block)
                        if len(where) == 1:
                            SE_val_list[e][where[0]][i,:,:,:] += val_block
                        elif len(where) == 0:
                            SE_val_list[e].append(np.zeros((nk, ) + val_block.shape, dtype = dtype))
                            SE_ind_list[e].append(ind_block)
                            SE_val_list[e][-1][i,:,:,:] += val_block
                        else:
                            print('\n Multiply counted block: Error, you messed up\n')
                            assert 1 == 0
            if nolowdin == False:
                esk, vsk = np.linalg.eigh(sk[piv, :][:, piv].toarray())
                s1_12   = vsk.dot(np.diag(1/np.sqrt(esk))).dot(vsk.T.conj())
                s1_12   = sp.csr_matrix(s1_12)
                s2_12   = vsk.dot(np.diag(  np.sqrt(esk))).dot(vsk.T.conj())
                s2_12   = sp.csr_matrix(s2_12)
                S1_i, S1_j, S1_v = sparse_find_faster(s1_12)
                S2_i, S2_j, S2_v = sparse_find_faster(s2_12)
                I_S1,V_S1 = Build_BS_purenp(S1_i, S1_j, S1_v, P)
                I_S2,V_S2 = Build_BS_purenp(S2_i, S2_j, S2_v, P)
                
                for count in range(len(I_S1)):
                    ind, val = I_S1[count], V_S1[count]
                    ind = (ind[0], ind[1])
                    L1_vals[L1_inds.index(ind)][i,0,:,:] += val
                for count in range(len(I_S2)):
                    ind, val = I_S2[count], V_S2[count]
                    ind = (ind[0], ind[1])
                    L2_vals[L2_inds.index(ind)][i,0,:,:] += val
                
                Lowdin =  [block_sparse(L1_inds, L1_vals, iGreens.Block_shape),
                           block_sparse(L2_inds, L2_vals, iGreens.Block_shape)
                          ]
                
            iH, jH, vH = sparse_find_faster(hk[piv,:][:,piv])
            iS, jS, vS = sparse_find_faster(sk[piv,:][:,piv])
            #print(sk[piv,:][:,piv][:5, :5])
            Ai, Bi, Ci = Build_BTD_purenp(iH, jH, vH,    nS)
            Ais, Bis, Cis = Build_BTD_purenp(iS, jS, vS, nS)
            
            for bb in range(n_diags):
                Hamiltonian.Al[bb][i,0,:,:] += Ai[bb]
                BTD_Overlap.Al[bb][i,0,:,:] += Ais[bb]
                if bb < n_diags -1:
                    Hamiltonian.Bl[bb][i,0,:,:] += Bi[bb]
                    Hamiltonian.Cl[bb][i,0,:,:] += Ci[bb]
                    BTD_Overlap.Bl[bb][i,0,:,:] += Bis[bb]
                    BTD_Overlap.Cl[bb][i,0,:,:] += Cis[bb]
        
        Gammas        = [block_sparse(SM_ind_list[e], SM_val_list[e], iGreens.Block_shape) for e in range(n_elecs)]
        self_energies = [block_sparse(SE_ind_list[e], SE_val_list[e], iGreens.Block_shape) for e in range(n_elecs)]
        if nolowdin==False:
            return iGreens, Gammas, Lowdin, Hamiltonian, BTD_Overlap, self_energies
        else:
            return iGreens, Gammas, Hamiltonian, BTD_Overlap, self_energies



def read_SE_from_tbtrans(filename, sort = False, dtype = np.complex64):
    i       = 0
    tbt     = sisl.get_sile(filename)
    SE_inds = []
    i       = 0
    while True:
        try:
            SE_inds += [list(tbt.pivot(elec = i, sort = sort))]
            i += 1
        except:
            break
    
    nk      =  tbt.nk
    ne      =  tbt.ne
    n_elec  =  len(SE_inds)
    SE      =  []
    
    for e in range(n_elec):
        n_idx = len(SE_inds[e])
        se = np.zeros((nk, ne, n_idx, n_idx), dtype = dtype)
        for i in range(nk):
            for j in range(ne):
                se[i,j, :, : ] = np.array(tbt.self_energy(elec = e, E = tbt.E[j], k = i, sort = sort))
        SE += [se.copy()]
    
    return SE, SE_inds

def read_overlap_data(tbt, SE_inds, Dev, Hd, Sd):
    geom      = tbt.geom
    kp, nk    = tbt.k, len(tbt.k)
    elec_inds = Dev.elec_inds
    elec_orbs = [np.hstack([np.arange(geom.a2o(i),geom.a2o(i+1)) 
                            for i in ei]) 
                 for ei in elec_inds]
    HS_elec   = [e.read_TSHS()  for e in Dev.elecs]
    rsse_mesg = '---> Hello! <--- \n' \
               +'It seems like you have used the a real-space self energy! good for you.\n'\
               +'This does however mean you need to compute a  rigid shift in the energy-levels in the Hamiltonian!\n'\
               +'This is done using the self.Renormalise_H and the find_correction from fitting tools.\n'\
               +'This should be covered in a tutorial.\n ----><----'
    orth_lead = []
    for elec in Dev.elecs:
        se_type = elec.which_type_SE()
        if se_type == 'normal':
            orth_lead += [ortho_lead(elec.read_TSHS(), elec.semi_inf)]
        elif se_type == 'RSSE':
            print(rsse_mesg)
            orth_lead += [ortho_lead_rsse(elec.read_minimal_TSHS(), 
                                          elec._ax_integrate,
                                          elec._ax_decimation,
                                          elec._supercell,
                                          elec._dk
                                          )]
        elif se_type == 'surfaceRSSE':
            print(rsse_mesg)
            orth_lead += [ortho_lead_sur_rsse(elec.read_minimal_TSHS(),
                                              elec.read_rssi_surface_TSHS(),
                                              elec.semi_inf,
                                              elec._ax_integrate,
                                              elec._supercell,
                                              elec._dk
                                             )]
        else:
            print('The 2E RSSE method is not currently implemented')
            assert 1 == 0
        
    
    Sig0 = [np.zeros((nk, len(sei), len(sei)), dtype=np.complex128) 
            for sei in SE_inds]
    Sig1 = [np.zeros((nk, len(sei), len(sei)), dtype=np.complex128) 
            for sei in SE_inds]
    for ik in range(nk):
        kpnt = kp[ik]
        Hdk = Hd.Hk(k = kpnt)
        Sdk = Sd.Sk(k = kpnt)
        for ie in range(len(Dev.elecs)):
            idx_D, idx_E = SE_inds[ie], elec_orbs[ie]
            H_D_l = Hdk[idx_D,:][:, idx_E].toarray()
            S_D_l = Sdk[idx_D,:][:, idx_E].toarray()
            Sinv_ll = orth_lead[ie].calc_Sinv(kpnt,  10)
            if orth_lead[ie].method.lower() == 'normal':
                print('Calculating corrections for electrode '+str(ie)+'. (Normal electrode)' )
                sig0=  S_D_l @ \
                       orth_lead[ie].converge_Sinv_Hl_Sinv(kpnt) @ \
                       (S_D_l.T.conj()) \
                     - S_D_l @ Sinv_ll @ (H_D_l.T.conj()) \
                     - H_D_l @ Sinv_ll @ (S_D_l.T.conj())
                
                sig1 = S_D_l @ Sinv_ll @ (S_D_l.conj().T)
            elif orth_lead[ie].method.lower() == 'rsse':
                print('Calculating corrections for electrode '+ str(ie) +'. (RSSE electrode)' )
                S_ll_inD = Sdk[idx_E,:][:, idx_E].toarray()
                sig1     = S_D_l @ np.linalg.inv(S_ll_inD - Sinv_ll) @ (S_D_l.T.conj())
                sig0     = np.zeros(sig1.shape, dtype=np.complex128)
            elif orth_lead[ie].method.lower() == 'surfacersse':
                print('Calculating corrections for electrode '+ str(ie)+ '. (surfaceRSSE electrode)' )
                S_ll_inD = Sdk[idx_E,:][:, idx_E].toarray()
                sig1     = S_D_l @ np.linalg.inv(S_ll_inD - Sinv_ll) @ (S_D_l.T.conj())
                sig0     = np.zeros(sig1.shape, dtype=np.complex128)
            
            Sig0[ie][ik,:,:] = sig0
            Sig1[ie][ik,:,:] = sig1
            
    return Sig0, Sig1

def pivot_and_sub(SigList, pivot, idxD, no):
    New1 = []
    for iE in range(len(SigList)):
        idx = np.array(idxD[iE])
        Sig    = SigList[iE]
        nk     = Sig.shape[0]
        outsig = np.zeros((nk, no, no), dtype=np.complex128)
        for ik in range(nk):
            outsig[ik, idx[:, None], idx[None,:]] = Sig[ik]
        New1.append(outsig)
    return [A[:,pivot,:][:,:,pivot].copy() for A in New1]

        
    
    
    
    
    

class ortho_lead:
    def __init__(self, HS, direc):
        Hm, Sm           = sisl2array_min(HS.transform(orthogonal=False))
        self.Hm, self.Sm = Hm, Sm
        self.HS          = HS
        self.no = Hm.shape[-1]
        self.direc = direc
        self.method = 'normal'
    def get_UC_inds(self):
        nx,ny,nz= self.HS.nsc
        Nx      = np.arange(-(nx//2), (nx//2)+1)
        Ny      = np.arange(-(ny//2), (ny//2)+1)
        Nz      = np.arange(-(nz//2), (nz//2)+1)
        return (nx, ny, nz), Nx, Ny, Nz
    def eval_HS_k(self, kvec):
        ax, d   = translate2(translate(self.direc))
        (nx,ny,nz), Nx, Ny, Nz = self.get_UC_inds()
        Hm, Sm  = self.Hm, self.Sm
        assert (Hm.shape[ax]==3 and Sm.shape[ax]==3)
        kv = [kvec[0],kvec[1],kvec[2]]
        kv.pop(ax); kv = np.array(kv)
        Nx_i = [Nx,Ny,Nz]; Nx_i.pop(ax)
        Nd1, Nd2 = Nx_i
        H00, H01 = Hm.take(0,axis=ax), Hm.take(d,axis=ax)
        S00, S01 = Sm.take(0,axis=ax), Sm.take(d,axis=ax)
        exp_ph  =  np.exp(2j*np.pi*
                          np.add.outer(kv[0]*Nd1, 
                                       kv[1]*Nd2 ))[:,:, None, None]
        def dosum(L):
            return [(exp_ph * l).sum(axis=(0,1)) for l in L]
        return dosum([H00, H01, S00, S01]) # Order is important here
    def construct_BTD(self, kvec, nblocks):
        h00, h01, s00, s01 = self.eval_HS_k(kvec)
        zero_mat = np.zeros(s00.shape, dtype=np.complex128)
        Sinv_sur = SE_1(1.0, zero_mat,zero_mat, s00, s01)
        Al = [s00 - Sinv_sur,  s00]
        Cl = [s01.T.conj()]
        Bl = [s01]
        Ial= [ 0, ] + [1]*(nblocks - 1)
        Ibl= [ 0, ]*(nblocks-1)
        Icl= [ 0, ]*(nblocks-1)
        btd= block_td(Al, Bl, Cl, Ial, Ibl, Icl)
        return btd, h00, h01
    def calc_Sinv_Hl_Sinv(self,kvec, nblocks):
        btd, h00, h01 = self.construct_BTD(kvec, nblocks)
        msk        = np.diag(np.ones(nblocks))
        msk[:, -1] = np.ones(nblocks)
        msk[-1, :] = np.ones(nblocks)
        ibtd       = btd.Invert(msk)
        res        = np.zeros(h00.shape, dtype = np.complex128)
        def Hblock(i,j):
            if i == j:   return h00
            if j+1 == i: return h01
            if j-1 == i: return h01.conj().T
            return None
        for k,l in itertools.product(range(nblocks), range(nblocks)):
            mat = Hblock(k, l)
            if mat is not None:
                res += ibtd.Block(nblocks-1, k) @ mat @ ibtd.Block(l, nblocks-1)
        return res
    def converge_Sinv_Hl_Sinv(self, kvec, tol = 1e-6, 
                              maxcount = 25, fail_on_break = True):
        nb   = 3
        diff = tol * 10.0
        count = 1
        while diff>tol:
            res1 = self.calc_Sinv_Hl_Sinv(kvec, nb-1)
            res2 = self.calc_Sinv_Hl_Sinv(kvec, nb)
            diff = np.abs(res2 - res1).sum()
            count += 1
            nb    += 1
            if count>maxcount:
                print('didnt converge the inverse of the overlap matrix :( ')
                if fail_on_break:
                    assert 1 == 0
                break
        return res2
    def calc_Sinv(self, kvec, nblocks):
        btd, h00, h01 = self.construct_BTD(kvec, nblocks)
        msk        = np.diag(np.ones(nblocks))
        ibtd = btd.Invert(msk)
        return ibtd.Block(nblocks-1, nblocks-1)

class ortho_lead_rsse:
    def __init__(self,   HS,
                 ax_int, ax_dec,  supercell,
                 dk):
        self.HS     = HS.transform(orthogonal=False).copy()
        self.ax_int = ax_int
        self.ax_dec = ax_dec
        self.dk     = dk
        self.supercell = supercell
        self.nullify_H(delta=1e-7)
        self.make_sisl_RSSE()
        self.method = 'RSSE'
    
    @staticmethod
    def NxNyNz(nsc):
        nx,ny,nz= nsc 
        Nx      = np.arange(-(nx//2), (nx//2)+1)
        Ny      = np.arange(-(ny//2), (ny//2)+1)
        Nz      = np.arange(-(nz//2), (nz//2)+1)
        return Nx, Ny, Nz
    
    def nullify_H(self, delta = 0.0):
        if hasattr(self, '_HS'):
            return
        Nx,Ny,Nz= self.NxNyNz(self.HS.nsc)
        self._HS= self.HS.copy()
        no      = self.HS.no
        for ix,iy,iz,oi,oj in itertools.product(Nx,Ny,Nz,range(no),range(no)):
            me = self.HS[oi, oj, (ix,iy,iz)]
            hij, sij = me[0]+0.0,me[1]+0.0
            hij = hij * 1e-9
            if abs(hij) + abs(sij) > 1e-9:
                self.HS[oi,oj,(ix,iy,iz)] = (hij, sij)
    
    def make_sisl_RSSE(self):
        self.RSSE =  sisl.RealSpaceSE(self.HS, self.ax_dec, 
                                      self.ax_int, self.supercell, 
                                      dk = self.dk)
    
    def calc_Sinv(self, dummy1, dummy2):
        se = self.RSSE.self_energy(1.0, coupling = True)
        self._outshape = se.shape
        return se
    
    def converge_Sinv_Hl_Sinv(self, dummy1):
        return np.zeros(self._outshape,dtype=np.complex128)

class ortho_lead_sur_rsse:
    def __init__(self,   HS, Hsurf, direc, 
                 ax_int, supercell,
                 dk):
        self.HS = HS.transform(orthogonal=False).copy()
        self.Hsurf = Hsurf.transform(orthogonal=False).copy()
        self.direc  = direc
        self.ax_int = ax_int
        self.dk     = dk
        self.supercell = supercell
        self.nullify_H(delta=1e-7)
        self.make_sisl_SurRSSE()
        self.method = 'surfaceRSSE'
    
    @staticmethod
    def NxNyNz(nsc):
        nx,ny,nz= nsc 
        Nx      = np.arange(-(nx//2), (nx//2)+1)
        Ny      = np.arange(-(ny//2), (ny//2)+1)
        Nz      = np.arange(-(nz//2), (nz//2)+1)
        return Nx, Ny, Nz
    
    def nullify_H(self, delta = 0.0):
        if hasattr(self, '_HS'):
            return
        Nx,Ny,Nz= self.NxNyNz(self.HS.nsc)
        self._HS= self.HS.copy()
        self._Hsurf = self.Hsurf.copy()
        no      = self.HS.no
        for ix,iy,iz,oi,oj in itertools.product(Nx,Ny,Nz,range(no),range(no)):
            me = self.HS[oi, oj, (ix,iy,iz)]
            hij, sij = me[0]+0.0,me[1]+0.0
            hij = hij * delta
            if abs(hij) + abs(sij) > 1e-9:
                self.HS[oi,oj,(ix,iy,iz)] = (hij, sij)
        
        Nx,Ny,Nz= self.NxNyNz(self.Hsurf.nsc)
        no      = self.Hsurf.no
        for ix,iy,iz,oi,oj in itertools.product(Nx,Ny,Nz,range(no),range(no)):
            me = self.Hsurf[oi, oj, (ix,iy,iz)]
            hij, sij = me[0]+0.0,me[1]+0.0
            hij = hij * delta
            if abs(hij) + abs(sij) > 1e-9:
                self.Hsurf[oi,oj,(ix,iy,iz)] = (hij, sij)
        
    def make_sisl_SurRSSE(self):
        self.SE    = sisl.RecursiveSI(self.HS, translate(self.direc))
        self.SRSSE = sisl.RealSpaceSI(self.SE, self.Hsurf, 
                                      self.ax_int, 
                                      unfold = self.supercell,
                                      dk     = self.dk)
    
    def calc_Sinv(self, dummy1, dummy2):
        se = self.SRSSE.self_energy(1.0, coupling = True)
        self._outshape = se.shape
        return se
    
    def converge_Sinv_Hl_Sinv(self, dummy1):
        return np.zeros(self._outshape,dtype=np.complex128)
    
def SE_1(z,H,V,S00,S01,eps=1e-10,DT=np.complex128):
    n          =  len(H)
    beta       =  CZ((n,n))
    igb        =  S00*z - H
    alpha      =  V - S01 * z
    beta [:,:] =  V.conj().T - S01.conj().T * z
    sse        =  CZ((n,n))
    while True:
        gb       = np.linalg.inv(igb)
        gb_beta  = np.dot(gb,beta)
        gb_alpha = np.dot(gb,alpha)
        arr1     = np.dot(alpha,gb_beta)
        sse     += arr1
        igb     -= arr1 + np.dot(beta,gb_alpha)
        alpha    = np.dot(alpha,gb_alpha)
        beta     = np.dot(beta,gb_beta)
        if (np.sum(np.abs(alpha))+np.sum(np.abs(beta)))<eps:
            return sse

def sisl2array_min(H):
    nx, ny, nz = H.nsc
    no    = H.no
    Hops  = np.zeros((nx,ny,nz, no, no), dtype = np.complex128)
    Sops  = np.zeros((nx,ny,nz, no, no), dtype = np.complex128)
    for I in range(-(nx//2), (nx//2)+1):
        for J in range(-(ny//2), (ny//2)+1):
            for K in range(-(nz//2), (nz//2)+1):
                for io in range(no):
                    for jo in range(no):
                        hijR = H[io,jo,(I,J,K)]
                        Hops[I,J,K,io,jo] = hijR[0]
                        Sops[I,J,K,io,jo] = hijR[1]
    return Hops, Sops

def translate(s):
    return s.replace('a1', 'A').replace('a2', 'B').replace('a3', 'C')

def translate2(s):
    if   '+' in s:  d =  1
    elif '-' in s:  d = -1
    if   'A' in s: ax =  0
    elif 'B' in s: ax =  1
    elif 'C' in s: ax =  2
    return ax, d
def HiThere(a, b,c): 
    return a+b+c






# def get_artf_obj(HS, semi_inf):
#     nx, ny, nz = HS.nsc
#     Hobj = HS.copy()
#     Sobj = HS.copy()
#     # make objects to calculate H_\alpha ^-1 & S_\alpha^-1
#     for i in range(HS._csr.shape[0]):
#         for j in range(HS._csr.shape[1]):
#             Sobj._csr[i,j,0] = 0.0
#             Hobj._csr[i,j,1] = 0.0
#     # SE_S = sisl.physics.RecursiveSI(Sobj, translate(semi_inf))
#     # SE_H = sisl.physics.RecursiveSI(Hobj, translate(semi_inf))
#     return Sobj, Hobj# , SE_S, SE_H



    
        
        
    
# def get_elec_overlap_inv(HS, semi_inf, kvec, nblocks):
#     ax, d   = translate2(translate(semi_inf))
#     Hm, Sm  = sisl2array_min(HS)
#     # return Hm, Sm
#     nx,ny,nz= HS.nsc
#     Nx      = np.arange(-(nx//2), (nx//2)+1)
#     Ny      = np.arange(-(ny//2), (ny//2)+1)
#     Nz      = np.arange(-(nz//2), (nz//2)+1)
#     if ax == 0:
#         assert Hm.shape[0] == 3
#         assert Sm.shape[0] == 3
#         H00, H01 = Hm[0], Hm[d]
#         S00, S01 = Sm[0], Sm[d]
#         Nd1,Nd2 = Ny, Nz
#         n1,n2   = ny, nz
#         kv = np.array([kvec[1],kvec[2]])
#     if ax == 1:
#         assert Hm.shape[1] == 3
#         assert Sm.shape[1] == 3
#         H00, H01 = Hm[:, 0], Hm[:, d]
#         S00, S01 = Sm[:, 0], Sm[:, d]
#         Nd1, Nd2 = Nx, Nz
#         n1,n2   = nx, nz
#         kv = np.array([kvec[0],kvec[2]])
#     if ax == 2:
#         assert Hm.shape[2] == 3
#         assert Sm.shape[2] == 3
#         H00, H01 = Hm[:, :, 0], Hm[:, :, d]
#         S00, S01 = Sm[:, :, 0], Sm[:, :, d]
#         Nd1, Nd2 = Nx, Ny
#         n1,n2    = nx, ny
#         kv = np.array([kvec[0],kvec[1]])
#     no = H00.shape[-1];  dt = np.complex128
#     h00 = np.zeros((no,no), dtype=dt)
#     h01 = np.zeros((no,no), dtype=dt)
#     s00 = np.zeros((no,no), dtype=dt)
#     s01 = np.zeros((no,no), dtype=dt)
#     zero_mat = np.zeros((no,no), dtype=dt)
#     for i1 in Nd1:
#         for i2 in Nd2:
#             phase  = kv[0]*i1 + kv[1]*i2
#             exp_ph = np.exp(2j*np.pi*phase)
#             h00 += H00[i1,i2,:,:] * exp_ph
#             h01 += H01[i1,i2,:,:] * exp_ph
#             s00 += S00[i1,i2,:,:] * exp_ph
#             s01 += S01[i1,i2,:,:] * exp_ph
    
#     Sinv_sur = SE_1(1.0, zero_mat,zero_mat, s00, s01)
#     Al = [s00 - Sinv_sur,  s00]
#     Cl = [s01.T.conj()]
#     Bl = [s01]
#     Ial= [ 0, ] + [1]*(nblocks - 1)
#     Ibl= [ 0, ]*(nblocks-1)
#     Icl= [ 0, ]*(nblocks-1)
#     btd= block_td(Al, Bl, Cl, Ial, Ibl, Icl)
#     #msk        = np.diag(np.ones(nblocks))
#     #msk[:, -1] = np.ones(nblocks)
#     #msk[-1, :] = np.ones(nblocks)
#     inv        = btd.Invert()#(msk)
#     return inv

    
    
    
    
    
    
    
    
    
    