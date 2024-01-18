import numpy as np
from math import exp
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
from rewards2 import get_init_dist, evaluate_mol, modify_fragment, evaluated_mols
import logging

from rewards import decode

from rdkit import Chem

import json

##################################################
############# Simulated Annealing ################

def decaimentoTemperatura(To, k, alpha):
    return To/(1 + alpha*k)

def getChancePassosIndireto(delta, t):
    return exp(-delta/t)

temperatura_inicial = 30
alpha     = 0.9

##################################################


scores = 1. / TIMES
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1



# Train actor and critic networks
def train(X, actor, critic, decodings, out_dir=None):

    mols       = {}
    dic_temp   = {}
    dic_delta  = {}
    dic_prob   = {}
    dic_accept = {}
    n_total = 0
    dist = get_init_dist(X, decodings)
    m = X.shape[1]
    with open('New_Mols.txt', 'w') as arquivo:
    # For every epoch
        for e in range(EPOCHS):
            # Select random starting "lead" molecules
            rand_n = np.random.randint(0,X.shape[0],BATCH_SIZE)
            batch_mol = X[rand_n].copy()
            # ---> Simulated Annealing <--- #            
            n_total = n_total + 1
            temp = decaimentoTemperatura(temperatura_inicial, n_total, alpha)
            dic_temp[e] = temp
            # ---> Simulated Annealing <--- #

            # For all modification steps
            for t in range(TIMES):

                tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES

                # Select actions
                for i in range(BATCH_SIZE):

                    a = np.random.randint(0, 62)

                    a = int(a // MAX_SWAP)

                    if a == 12:
                        a = 11

                    s = a % MAX_SWAP                                        
                    mol_orriginal = batch_mol[i,a]
                    mol_orriginal_av = batch_mol[i]
                    batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)                    
                    fr, score = evaluate_mol(batch_mol[i], e, decodings)                    
                    if all(fr):
                        mol_new = decode(batch_mol[i], decodings)
                        smiles_code = Chem.MolToSmiles(mol_new)
                        new = mols.get(smiles_code, 0)
                        if new == 0:
                            mols[smiles_code] =  1
                            arquivo.write(f'{smiles_code}\n')
                        else:
                            mols[smiles_code] =  new + 1                            
                        #print('Uma molecula atendeu')
                    # Colocar uma prob aqui do simulated anealing - ok
                    # Aqui proximo Sprint
                    # melhorar a perfromance verificar se a temperatura for zero
                    # Nao salvar duplicado tentar salar o JSON do rewards evaluated_mols
                    else:
                        _, score_old = evaluate_mol(mol_orriginal_av, e, decodings)                    
                        delta = (score_old - score)
                        if delta > 0:
                            chancePassoIndireto = getChancePassosIndireto(delta, temp)
                            dic_delta[e] = delta
                            dic_prob[e] = chancePassoIndireto
                            if not(np.random.rand()<=chancePassoIndireto):
                                batch_mol[i,a] = mol_orriginal
                                dic_accept[e] = 1
                
            # np.save("History/out-{}.npy".format(e), batch_mol)

            print (f"Epoch {e}")

    with open("temperaturas.json", 'w') as arquivo:
        json.dump(dic_temp, arquivo)    

    with open("deltas.json", 'w') as arquivo:
        json.dump(dic_delta, arquivo)            

    with open("prob.json", 'w') as arquivo:
        json.dump(dic_prob, arquivo)        

    with open("aceitas.json", 'w') as arquivo:
        json.dump(dic_accept, arquivo)        

    return True
