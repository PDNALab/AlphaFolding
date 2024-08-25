import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import Bio.PDB as bp
from Bio.PDB import *
import os, re
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import protein, residue_constants
from colabdesign.shared.protein import _np_get_cb
import pickle
from colabdesign import af
import numpy as np
import jax.numpy as jnp
import jax
from scipy.special import softmax
import sys
import tqdm.notebook
import argparse


def parse_args():                             
    parser = argparse.ArgumentParser()
    parser.add_argument('-seq',type=str,default='',help='input sequence')
    parser.add_argument('-recycle',type=int,help='number of recycles')
    parser.add_argument('-iter',type=int,help='number of iterations')
    parser.add_argument('-seqsep',type=int,default=0,help='sequence separation mask length')
    parser.add_argument('-noise',type=str,default='g',help='noise type gumbel (g) or uniform (u)')
    parser.add_argument('-sample_model',action='store_true')
    parser.add_argument('-tm_seq',action='store_true',help='sequence separation mask length')
    parser.add_argument('-seq_n',action='store_true',help='use sequence noise or not')
    parser.add_argument('-dgram_n',action='store_true',help='use distogram noise or not')
    parser.add_argument('-dropout',action='store_true',help='use dropout')
    parser.add_argument('-temp_mode',type=str,default=None,help='mode of how to feed predicted structure in the next iteration')
    parser.add_argument('-model_name',type=str,default=None,help='model name')  
    parser.add_argument('-template_path',default=None,type=str,help='use template or not')
    return parser.parse_args()
 
args=parse_args()

def get_dgram(positions, num_bins=39, min_bin=3.25, max_bin=50.75):
  atom_idx = residue_constants.atom_order
  atoms = {k:positions[...,atom_idx[k],:] for k in ["N","CA","C"]}
  cb = _np_get_cb(**atoms, use_jax=False)
  dist2 = np.square(cb[None,:] - cb[:,None]).sum(-1,keepdims=True)
  lower_breaks = np.linspace(min_bin, max_bin, num_bins)
  lower_breaks = np.square(lower_breaks)
  upper_breaks = np.concatenate([lower_breaks[1:],np.array([1e8], dtype=jnp.float32)], axis=-1)
  return ((dist2 > lower_breaks) * (dist2 < upper_breaks)).astype(float)

def sample_gumbel(shape, eps=1e-10): 
  """Sample from Gumbel(0, 1)"""
  U = np.random.uniform(size=shape)
  return -np.log(-np.log(U + eps) + eps)

def sample_uniform(shape, eps=1e-10): 
  """Sample from Uniform(0, 1)"""
  U = np.random.uniform(size=shape)
  return U + eps

from colabdesign.af.alphafold.common import residue_constants
def xyz_atom37(pdb_file):
  """
  Convert atom coordinates [num_atom, 3] from xyz read from file such as pdb to atom37 format.
  """
  atom37_order = residue_constants.atom_order
  parser = PDBParser()
  structure = parser.get_structure("A", pdb_file)
  atoms = list(structure.get_atoms())
  length = len(list(structure.get_residues()))
  atom37_coord = np.zeros((length, 37, 3))
  
  for atom in atoms:
    atom37_index = atom37_order[atom.get_name()]
    residue_index = atom.get_parent().id[1]
    atom37_coord[residue_index-1][atom37_index] = atom.get_coord()
  return atom37_coord

starting_seq = args.seq 
starting_seq = re.sub("[^A-Z]", "", starting_seq.upper())
length = len(starting_seq)
print(f"run seq {args.seq} with recycle {args.recycle} and dropout {args.dropout}")
aa_order = residue_constants.restype_order
start_seq_aatype=np.array([aa_order[i] for i in starting_seq])
model_names=None if not args.model_name else [args.model_name]
use_multimer = False 
mode = "dgram_retrain" if not args.temp_mode else args.temp_mode
template_path = args.template_path


if template_path != None:
  template_path = os.path.join(os.getcwd(),template_path)

clear_mem()
af_model = mk_afdesign_model(protocol="hallucination",
                             use_templates=True,
                             debug=True, 
                             model_names=model_names,
                             use_multimer=use_multimer)
af_model.prep_inputs(length=length)

if "dgram" in mode:
  if "retrain" in mode and not use_multimer:
    # update distogram head to return all 39 bins
    af_model._cfg.model.heads.distogram.first_break = 3.25
    af_model._cfg.model.heads.distogram.last_break = 50.75
    af_model._cfg.model.heads.distogram.num_bins = 39
    af_model._model = af_model._get_model(af_model._cfg)
    from colabdesign.af.weights import __file__ as af_path
    template_dgram_head = np.load(os.path.join(os.path.dirname(af_path),'template_dgram_head.npy'))
    for k in range(len(af_model._model_params)):
      params = {"weights":jnp.array(template_dgram_head[k]),"bias":jnp.zeros(39)}
      af_model._model_params[k]["alphafold/alphafold_iteration/distogram_head/half_logits"] = params
  else:
    dgram_map = np.eye(39)[np.repeat(np.append(0,np.arange(15)),4)]
    dgram_map[-1,:] = 0 

#set up template
iterations = args.iter 
use_dgram_noise = args.dgram_n 
use_seq_noise = args.seq_n 
use_dropout = args.dropout 
seqsep_mask =  args.seqsep 

#set up AlphaFold 
sample_models = args.sample_model 
num_recycles = args.recycle 

L = sum(af_model._lengths)
af_model.restart(mode="gumbel")
#af_model._inputs["rm_template_seq"] = args.tm_seq

# gather info about inputs
if "offset" in af_model._inputs:
  offset = af_model._inputs
else:
  idx = af_model._inputs["residue_index"]
  offset = idx[:,None] - idx[None,:]

# initialize sequence
if len(starting_seq) > 1:
  af_model.set_seq(seq=starting_seq)
af_model._inputs["bias"] = np.zeros((L,20))

# initialize coordinates/dgram
af_model._inputs["batch"] = {"aatype":np.zeros(L).astype(int),
                             "all_atom_mask":np.zeros((L,37)),
                             "all_atom_positions":np.zeros((L,37,3)),
                             "dgram":np.zeros((L,L,39))}
if template_path is not None:
  xyz = xyz_atom37(pdb_file=template_path)
  af_model._inputs["batch"]=af.prep.prep_pdb(template_path)['batch']
  af_model._inputs["batch"]["all_atom_positions"] = xyz
  p = PDBParser()   
  structure = p.get_structure("X", template_path)
  plddts=np.array([structure[0]["A"][i+1]["CA"].get_bfactor() for i in range(len(structure[0]['A']))])
  af_model._inputs["batch"]['all_atom_mask'][:,:4]=np.sqrt(plddts/100)[:,None]
  dgram = get_dgram(xyz)
  mask = np.abs(offset) > seqsep_mask
  af_model._inputs["batch"]["dgram"] = dgram * mask[:,:,None]
  if use_dgram_noise:
    if dgram_noise_type == "g":   
      noise = sample_gumbel(dgram.shape) * (1 - k/iterations)
      dgram = softmax(np.log(dgram + 1e-8) + noise, -1)
    elif dgram_noise_type == 'u':  
      noise = sample_uniform(dgram.shape) * (1 - k/iterations)
      dgram = softmax(np.log(dgram + 1e-8) + noise, -1)

#run iterative structure predictions
print(f'use seq, distogram noise, sample_model, template, mode: {use_seq_noise, use_dgram_noise, sample_models, (template_path is not None), mode}')
for k in range(iterations):
  if k > 0:
    dgram_xyz = get_dgram(xyz)
    dgram_prob = softmax(dgram_logits,-1)

    if mode == "xyz":
      dgram = dgram_xyz
    if mode == "dgram":
      dgram = dgram_prob @ dgram_map
      dgram[...,14:] = dgram_xyz[...,14:] * dgram_prob[...,-1:]
    if mode == "dgram_retrain":
      dgram = dgram_prob
    
    if use_dgram_noise:
      if dgram_noise_type == "g":   
        noise = sample_gumbel(dgram.shape) * (1 - k/iterations)
        dgram = softmax(np.log(dgram + 1e-8) + noise, -1)
      elif dgram_noise_type == 'u':  
        noise = sample_uniform(dgram.shape) * (1 - k/iterations)
        dgram = softmax(np.log(dgram + 1e-8) + noise, -1)

    # add mask to avoid local contacts being fixed (otherwise there is a bias toward helix), default is 0 since we'd like to use all structure informaion from last step
    mask = np.abs(offset) > seqsep_mask
    af_model._inputs["batch"]["dgram"] = dgram * mask[:,:,None]

  # denoise
  aux = af_model.predict(return_aux=True, verbose=True,
                        sample_models=sample_models,
                        dropout=use_dropout, num_recycles=num_recycles)
  plddt = aux["plddt"]
  #print(f"plddt shape: {af_model._inputs['batch']['all_atom_mask'].shape},{plddt.shape,np.sqrt(plddt)[:,None].shape}")
  seq = aux["seq"]["hard"][0].argmax(-1)
  xyz = aux["atom_positions"].copy()
  #print(f"xyz: {xyz.shape}")
  dgram_logits = aux["debug"]["outputs"]["distogram"]["logits"] 
  #print(f"seq: {seq}") 
  # update inputs    
  af_model._inputs["batch"]["aatype"] = seq
  af_model._inputs["batch"]["all_atom_mask"][:,:4] = np.sqrt(plddt)[:,None]
  af_model._inputs["batch"]["all_atom_positions"] = xyz

  # save results
  af_model._save_results(aux)
  af_model._k += 1
  af_model.save_pdb(f"iter_{k}.pdb")
