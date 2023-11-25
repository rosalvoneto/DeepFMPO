import numpy as np
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
from rewards import get_init_dist, evaluate_mol, modify_fragment
import logging


scores = 1. / TIMES
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1



# Train actor and critic networks
def train(X, actor, critic, decodings, out_dir=None):

    hist = []
    dist = get_init_dist(X, decodings)
    m = X.shape[1]

    # For every epoch
    for e in range(EPOCHS):
        # Select random starting "lead" molecules
        rand_n = np.random.randint(0,X.shape[0],BATCH_SIZE)
        batch_mol = X[rand_n].copy()
        
        # For all modification steps
        for t in range(TIMES):

            tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES

            # Select actions
            for i in range(BATCH_SIZE):

                a = np.random.randint(0, 62)

                a = int(a // MAX_SWAP)

                s = a % MAX_SWAP
                
                # Colocar uma prob aqui do simulated anealing
                mol_orriginal = batch_mol[i,a]
                batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)
                fr = evaluate_mol(batch_mol[i], e, decodings)
                if all(fr):
                    print('Uma molecula atendeu')
                else:
                    batch_mol[i,a] = mol_orriginal
            
        # np.save("History/out-{}.npy".format(e), batch_mol)

        print (f"Epoch {e}")
        

    return True
