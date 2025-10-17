import pickle

#access to variables file in different seeds
#take avarage of variables
#plot it in a new folder

# Load all variables
loaded_data = pickle.load(open('/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d32_heads2_patch2_sample1024_lr0.01_iter1000_symmTrue/J=0.5_L=4/variables.pkl', 'rb'))

# Access individual variables
#sign_vstate_MCMC = loaded_data['sign_vstate_MCMC']
sign_vstate_full = loaded_data['sign_vstate_full']
sign_exact = loaded_data['sign_exact']
fidelity = loaded_data['fidelity']
configs = loaded_data['configs']
sign_vstate_config = loaded_data['sign_vstate_config']
weight_exact = loaded_data['weight_exact']
weight_vstate = loaded_data['weight_vstate']
error = loaded_data['error']