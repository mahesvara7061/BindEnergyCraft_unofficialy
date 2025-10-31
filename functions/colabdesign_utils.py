####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .pyrosetta_utils import pr_relax, align_pdbs
from .generic_utils import update_failures

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])
    
    """
    Quick Diagnostic Script for Zero multi_ptme Issue
    ==================================================

    PASTE THIS CODE IMMEDIATELY AFTER af_model.prep_inputs() in your notebook
    (around line 38-48 in colabdesign_utils.py, in binder_hallucination function)
    """

    # ==================== DIAGNOSTIC CODE START ====================
    import numpy as np
    import jax.numpy as jnp

    print("\n" + "="*70)
    print("🔍 DUAL pTME DIAGNOSTIC REPORT")
    print("="*70)

    # 1. Check if inputs are prepared
    print("\n1️⃣ Checking af_model._inputs...")
    if hasattr(af_model, '_inputs'):
        inputs = af_model._inputs
        print(f"   ✅ _inputs exists with {len(inputs)} keys")
        print(f"   Keys: {list(inputs.keys())}")
    else:
        print(f"   ❌ _inputs NOT FOUND - this is a problem!")
        inputs = None

    # 2. Check for chain_index
    print("\n2️⃣ Checking chain_index...")
    if inputs and 'chain_index' in inputs:
        ci = np.array(inputs['chain_index'])
        unique, counts = np.unique(ci, return_counts=True)
        print(f"   ✅ chain_index FOUND!")
        print(f"   Unique chain IDs: {unique}")
        print(f"   Residue counts per chain: {counts}")
        print(f"   Total residues: {len(ci)}")
        print(f"   First 10 values: {ci[:10]}")
        print(f"   Last 10 values: {ci[-10:]}")
        
        # Diagnose the structure
        print("\n   📊 Structure Analysis:")
        if len(unique) == 2:
            print(f"   ⚠️  Found 2 chains (expected 3 for dual-target)")
            print(f"      → Chain 0: {counts[0]} residues (probably both targets merged)")
            print(f"      → Chain 1: {counts[1]} residues (probably binder)")
        elif len(unique) == 3:
            print(f"   ✅ Found 3 chains (good!)")
            print(f"      → Chain 0: {counts[0]} residues")
            print(f"      → Chain 1: {counts[1]} residues")
            print(f"      → Chain 2: {counts[2]} residues")
        else:
            print(f"   ⚠️  Found {len(unique)} chains (unexpected)")
    else:
        print(f"   ❌ chain_index NOT FOUND")
        if inputs:
            print(f"   Available input keys: {list(inputs.keys())}")
            # Try alternative keys
            for alt_key in ['asym_id', 'asym_index', 'entity_id']:
                if alt_key in inputs:
                    print(f"   ℹ️  Found alternative: '{alt_key}'")
                    alt_ci = np.array(inputs[alt_key])
                    unique_alt = np.unique(alt_ci)
                    print(f"      Unique values: {unique_alt}")

    # 3. Check target and binder lengths
    print("\n3️⃣ Checking model structure...")
    if hasattr(af_model, '_target_len'):
        print(f"   ✅ _target_len: {af_model._target_len}")
    else:
        print(f"   ❌ _target_len NOT SET")

    if hasattr(af_model, '_binder_len'):
        print(f"   ✅ _binder_len: {af_model._binder_len}")
    else:
        print(f"   ❌ _binder_len NOT SET")

    # 4. Check your dual pTME configuration
    print("\n4️⃣ Checking your dual pTME configuration...")
    print(f"   Config chains_A: {advanced_settings.get('dual_ptme_chains_A', 'NOT SET')}")
    print(f"   Config chains_B: {advanced_settings.get('dual_ptme_chains_B', 'NOT SET')}")
    print(f"   Config binder_chain_id: {advanced_settings.get('dual_ptme_binder_chain_id', 'NOT SET')}")

    # 5. Analyze compatibility
    print("\n5️⃣ Configuration vs Reality Check...")
    if inputs and 'chain_index' in inputs:
        ci = np.array(inputs['chain_index'])
        unique_chains = np.unique(ci)
        
        config_A = advanced_settings.get('dual_ptme_chains_A', [0])
        config_B = advanced_settings.get('dual_ptme_chains_B', [1])
        config_binder = advanced_settings.get('dual_ptme_binder_chain_id', 2)
        
        print(f"   Expected chains in config: {set(config_A + config_B + [config_binder])}")
        print(f"   Actual chains in structure: {set(unique_chains)}")
        
        missing = set(config_A + config_B + [config_binder]) - set(unique_chains)
        if missing:
            print(f"   ❌ MISMATCH! Missing chains: {missing}")
            print(f"   This is why multi_ptme is zero!")
        else:
            print(f"   ✅ All configured chains exist")

    # 6. Proposed solution
    print("\n6️⃣ RECOMMENDED SOLUTION:")
    if inputs and 'chain_index' in inputs:
        ci = np.array(inputs['chain_index'])
        unique, counts = np.unique(ci, return_counts=True)
        
        if len(unique) == 2:
            print("   Your structure has 2 chains (targets merged + binder)")
            print("   Solution A: Use residue-based dual pTME")
            print("   -------------------------------------------")
            print("   # In your colabdesign_utils.py, replace add_dual_ptme_softmax_loss call with:")
            print("   add_dual_ptme_residue_based(")
            print("       af_model,")
            if hasattr(af_model, '_target_len'):
                target_len = af_model._target_len
                # Assuming your dual_target.pdb: Chain A=192, Chain B=143
                # This is from your notebook output
                print(f"       target_A_start=0,")
                print(f"       target_A_end=191,      # Adjust based on your actual PDB")
                print(f"       target_B_start=192,")
                print(f"       target_B_end={target_len-1},")
            else:
                print(f"       target_A_start=0,")
                print(f"       target_A_end=???,      # YOUR CHAIN A LENGTH - 1")
                print(f"       target_B_start=???,    # CHAIN A LENGTH")
                print(f"       target_B_end=???,      # TOTAL TARGET LENGTH - 1")
            print("       weight=0.05,")
            print("       tau=0.2")
            print("   )")
            print()
            print("   OR")
            print()
            print("   Solution B: Update JSON config to match reality")
            print("   -----------------------------------------------")
            print("   {")
            print(f'       "dual_ptme_chains_A": [0],    // Target (merged)')
            print(f'       "dual_ptme_chains_B": [0],    // Same chain!')
            print(f'       "dual_ptme_binder_chain_id": {unique[-1]},  // Binder')
            print("       // Then modify loss function to use residue ranges")
            print("   }")
            
        elif len(unique) == 3:
            print("   Your structure has 3 chains - config should work!")
            print("   Current config:")
            print(f"   {{")
            print(f'       "dual_ptme_chains_A": {config_A},')
            print(f'       "dual_ptme_chains_B": {config_B},')
            print(f'       "dual_ptme_binder_chain_id": {config_binder}')
            print(f"   }}")
            print()
            print("   Suggested config based on structure:")
            print(f"   {{")
            print(f'       "dual_ptme_chains_A": [{unique[0]}],')
            print(f'       "dual_ptme_chains_B": [{unique[1]}],')
            print(f'       "dual_ptme_binder_chain_id": {unique[2]}')
            print(f"   }}")
    else:
        print("   ❌ Cannot determine solution - chain_index not available")
        print("   You may need to check:")
        print("   1. Is use_multimer_design: true in your config?")
        print("   2. Is prep_inputs being called correctly?")
        print("   3. Try running af_model._prep_features() manually")

    print("\n" + "="*70)
    print("🔍 END OF DIAGNOSTIC REPORT")
    print("="*70 + "\n")

    # ==================== DIAGNOSTIC CODE END ====================

    """
    WHAT TO DO NEXT:
    ================

    1. Copy the diagnostic code above
    2. Paste it in your notebook AFTER line 38 (after af_model.prep_inputs)
    3. Run ONE iteration of your design
    4. Read the diagnostic output carefully
    5. Apply the recommended solution

    The diagnostic will tell you EXACTLY what's wrong and how to fix it!
    """

        # DEBUG: xem các chain_index hiện có và số residue mỗi chain
    try:
        ci = af_model._inputs.get("chain_index", None)
        if ci is not None:
            uniq, counts = np.unique(np.array(ci), return_counts=True)
            print("[DEBUG] chain_index uniq:", uniq, "counts:", counts)
    except Exception as _e:
        pass

    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    "weights_ptme":advanced_settings["weights_ptme"]
                                    })

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
        

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    if advanced_settings["use_ptme_loss"]:   
        add_ptme_loss(af_model, advanced_settings["weights_ptme"]) 

        # ---- Dual-target pTME with softmax-balance (A vs B) ----
    if advanced_settings.get("use_dual_ptme_loss", False):
        # đọc cấu hình; nếu thiếu sẽ raise để bạn chủ động set
        chains_A = advanced_settings["dual_ptme_chains_A"]          # ví dụ: [0]
        chains_B = advanced_settings["dual_ptme_chains_B"]          # ví dụ: [1]
        binder_chain_id = advanced_settings["dual_ptme_binder_chain_id"]  # ví dụ: 2

        weight = float(advanced_settings.get("weights_multi_ptme", 0.05))
        tau = float(advanced_settings.get("tau_multi_ptme", 0.2))
        iface_thresh = float(advanced_settings.get("iface_thresh", 0.30))  # dùng ở bước 3

        add_dual_ptme_softmax_loss(
            af_model,
            chains_A=chains_A,
            chains_B=chains_B,
            binder_chain_id=binder_chain_id,
            weight=weight,
            tau_init=float(advanced_settings.get("tau_multi_ptme_init", 0.5)),
            tau_final=float(advanced_settings.get("tau_multi_ptme_final", 0.1)),
            iface_thresh=iface_thresh
        )


        # >>> THÊM GỌI PHẦN NÀY <<<
        add_dual_overlap_geodesic_losses(
            af_model,
            chains_A=chains_A,
            chains_B=chains_B,
            binder_chain_id=binder_chain_id,
            weight_overlap=float(advanced_settings.get("weights_overlap", 0.2)),
            weight_geo=float(advanced_settings.get("weights_geo", 0.1)),
            iface_thresh=float(advanced_settings.get("iface_thresh", 0.30)),
            r_cut=float(advanced_settings.get("geo_r_cut", 8.0)),
            sigma=float(advanced_settings.get("geo_sigma", 3.0)),
            geo_min=float(advanced_settings.get("geo_min", 25.0)),
        )

 

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        # uses gradient descend to get a PSSM profile and then uses PSSM to bias the sampling of random mutations to decrease loss
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        # 3 stage design using logits, softmax, and one hot encoding
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"], 
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        # design by using random mutations that decrease loss
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        # initial logits to prescreen trajectory
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

        # determine pLDDT of best iteration according to lowest 'loss' value
        initial_plddt = get_best_plddt(af_model, length)
        
        # if best iteration has high enough confidence then continue
        if initial_plddt > 0.65:
            print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                # temporarily dump model to assess secondary structure
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                # if beta sheeted trajectory is detected then choose to optimise
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            # how many logit iterations left
            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
            else:
                logit_plddt = initial_plddt

            # perform softmax trajectory design
            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # perform one hot encoding
            if softmax_plddt > 0.65:
                print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > 0.65:
                    # perform greedy mutation optimisation
                    print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                    if advanced_settings["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))

    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    ### save trajectory PDB
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # check clashes
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 or ca_clashes > 0:
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        # check if low quality prediction
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, failure_csv, seed=None):
    prediction_stats = {}

    try:
        if advanced_settings.get("use_ptme_loss", False):
            names = [getattr(cb, "__name__", "") for cb in prediction_model._callbacks["model"]["loss"]]
            if "loss_ptme" not in names:
                add_ptme_loss(prediction_model, weight=0.0)
    except Exception:
        pass

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}

    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # predict model
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'i_pTM': round(prediction_metrics['i_ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2),
                'i_pAE': round(prediction_metrics['i_pae'], 2),
            }
            ptme_val = prediction_metrics.get("ptme", None)
            stats["pTME"] = round(ptme_val, 2) if (ptme_val is not None) else None

            prediction_stats[model_num+1] = stats

            # List of filter conditions and corresponding keys
            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]

            # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

            if not pass_af2_filters:
                break

    # Update the CSV file with the failure counts
    if filter_failures:
        update_failures(failure_csv, filter_failures)

    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            pr_relax(complex_pdb, mpnn_relaxed)
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)

    return prediction_stats, pass_af2_filters

# run prediction for binder alone
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None):
    binder_stats = {}

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)

    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae

            # align binder model to trajectory binder
            align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats

    return binder_stats

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        print("Fixing interface residues: "+trajectory_interface_residues)
    else:
        fixed_positions = 'A'

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# Define pTMEnergy (pTME) loss for colabdesign / BindCraft
def add_ptme_loss(self, weight=0.05):
    """
    Add pTMEnergy (BECraft) as a differentiable loss term.
    - Uses AF 'predicted_aligned_error' logits and bin breaks.
    - Applies TM-style kernel g(d) = 1 / (1 + (d/d0(N))^2).
    - Averages over inter-chain (binder <-> target) pairs only.
    - Returns a scalar "ptme" to be MINIMIZED (more negative -> better interface).
    """

    def _tm_d0(N: int) -> jnp.ndarray:
        # classic TM-score d0(N), clamped to avoid tiny/negative values
        d0 = 1.24 * (jnp.maximum(jnp.asarray(N, jnp.float32), 1.0) - 15.0) ** (1.0 / 3.0) - 1.8
        return jnp.maximum(d0, jnp.asarray(1e-3, jnp.float32))

    def _bin_centers_from_breaks(breaks: jnp.ndarray | None, num_bins: int, max_err: float = 31.75) -> jnp.ndarray:
        if breaks is None:
            edges = jnp.linspace(0.0, max_err, num_bins + 1, dtype=jnp.float32)
        else:
            edges = jnp.concatenate(
                [jnp.array([0.0], jnp.float32), breaks.astype(jnp.float32), jnp.array([max_err], jnp.float32)], axis=0
            )
        return 0.5 * (edges[:-1] + edges[1:])  # (B,)

    def _masked_mean(x: jnp.ndarray, mask_bool: jnp.ndarray) -> jnp.ndarray:
        mask = mask_bool.astype(x.dtype)
        denom = jnp.maximum(mask.sum(), jnp.asarray(1.0, x.dtype))
        return (x * mask).sum() / denom

    def loss_ptme(inputs, outputs):
        pae = outputs.get("predicted_aligned_error", None)
        if pae is None or ("logits" not in pae):
            # If PAE logits are missing (e.g., wrong model), return a harmless zero
            return {"ptme": jnp.asarray(0.0, jnp.float32)}

        logits = pae["logits"]  # (L, L, B)
        L = logits.shape[0]
        B = logits.shape[-1]
        breaks = pae.get("breaks", None)

        # Build TM kernel over PAE bin centers
        centers = _bin_centers_from_breaks(breaks, B)                # (B,)
        d0 = _tm_d0(L)
        g = 1.0 / (1.0 + (centers / d0) ** 2)                        # (B,)

        logZ = logsumexp(logits, axis=-1)
        lse = logsumexp(logits + jnp.log(g + 1e-12)[None, None, :], axis=-1) - logZ

        # Inter-chain mask (target first, binder second)
        L_t = int(getattr(self, "_target_len", 0))
        L_b = int(getattr(self, "_binder_len", 0))
        if (L_t + L_b == L) and (L_t > 0 and L_b > 0):
            binder_vec = jnp.concatenate([jnp.zeros((L_t,), dtype=bool),
                                          jnp.ones((L_b,), dtype=bool)], axis=0)               # (L,)
            target_vec = jnp.logical_not(binder_vec)                                            # (L,)
            inter = jnp.logical_or(jnp.logical_and(binder_vec[:, None], target_vec[None, :]),
                                   jnp.logical_and(target_vec[:, None], binder_vec[None, :]))   # (L,L) bool
            ptme_energy = -_masked_mean(lse, inter)  # <-- no x[mask]; keeps static shape
        else:
            # fallback: try asym ids from inputs; if missing, average all pairs
            asym = None
            for k in ("asym_id", "chain_index", "asym_index"):
                if (inputs is not None) and (k in inputs):
                    asym = inputs[k]; break
            if (asym is not None) and (asym.shape[0] == L):
                inter = (asym[:, None] != asym[None, :])
                ptme_energy = -_masked_mean(lse, inter)
            else:
                ptme_energy = -jnp.mean(lse)

        return {"ptme": ptme_energy}

    # Register the loss callback and weight
    self._callbacks["model"]["loss"].append(loss_ptme)
    self.opt["weights"]["ptme"] = weight

# ---- Dual-target pTME with softmax-balance (A vs B) ----
def add_dual_ptme_softmax_loss(self, chains_A, chains_B, binder_chain_id, 
                                weight=0.05, tau_init=0.5, tau_final=0.1, 
                                iface_thresh=0.30):
    """
    Tính pTME riêng cho interface Binder<->A và Binder<->B, rồi cân bằng bằng softmax:
      L = weight * ( wA * pTME_A + wB * pTME_B ),  w = softmax([pTME_A, pTME_B]/tau)
    GHI CHÚ:
      - Chưa thêm geodesic/overlap ở bước này. Chỉ 1 scalar 'multi_ptme' để MINIMIZE.
      - Cách tính pTME dùng lại logic TM-kernel trên PAE logits như add_ptme_loss.
    """

    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    def _tm_d0(N: int) -> jnp.ndarray:
        d0 = 1.24 * (jnp.maximum(jnp.asarray(N, jnp.float32), 1.0) - 15.0) ** (1.0 / 3.0) - 1.8
        return jnp.maximum(d0, jnp.asarray(1e-3, jnp.float32))

    def _bin_centers_from_breaks(breaks: jnp.ndarray | None, num_bins: int, max_err: float = 31.75) -> jnp.ndarray:
        if breaks is None:
            edges = jnp.linspace(0.0, max_err, num_bins + 1, dtype=jnp.float32)
        else:
            edges = jnp.concatenate(
                [jnp.array([0.0], jnp.float32), breaks.astype(jnp.float32), jnp.array([max_err], jnp.float32)], axis=0
            )
        return 0.5 * (edges[:-1] + edges[1:])  # (B,)

    def _masked_mean(x: jnp.ndarray, mask_bool: jnp.ndarray) -> jnp.ndarray:
        mask = mask_bool.astype(x.dtype)
        denom = jnp.maximum(mask.sum(), jnp.asarray(1.0, x.dtype))
        return (x * mask).sum() / denom

    def _tm_lse_from_pae(pae):
        logits = pae["logits"]  # (L,L,B)
        L = logits.shape[0]
        B = logits.shape[-1]
        breaks = pae.get("breaks", None)
        centers = _bin_centers_from_breaks(breaks, B)     # (B,)
        d0 = _tm_d0(L)
        g = 1.0 / (1.0 + (centers / d0) ** 2)            # (B,)
        logZ = logsumexp(logits, axis=-1)
        lse = logsumexp(logits + jnp.log(g + 1e-12)[None, None, :], axis=-1) - logZ  # (L,L)
        return lse

    def _inter_mask(chain_idx, src_list, dst_id):
        # true ở (i,j) nếu i thuộc src_list và j==dst_id, hoặc ngược lại
        src = jnp.isin(chain_idx, jnp.array(src_list))
        dst = (chain_idx == dst_id)
        return jnp.logical_or(jnp.logical_and(src[:, None], dst[None, :]),
                              jnp.logical_and(dst[:, None], src[None, :]))  # (L,L) bool

    def _ptme_from_mask(lse, mask_bool):
        return -_masked_mean(lse, mask_bool)   # âm trung bình log-expected kernel → minimize

    def loss_dual_ptme(inputs, outputs):
        pae = outputs.get("predicted_aligned_error", None)
        if pae is None or ("logits" not in pae) or ("chain_index" not in inputs):
            return {"multi_ptme": jnp.asarray(0.0, jnp.float32)}

        lse = _tm_lse_from_pae(pae)                   # (L,L)
        chain_idx = inputs["chain_index"]             # (L,)

        mA = _inter_mask(chain_idx, jnp.array(chains_A), binder_chain_id)
        mB = _inter_mask(chain_idx, jnp.array(chains_B), binder_chain_id)

        pA = _ptme_from_mask(lse, mA)                 # scalar
        pB = _ptme_from_mask(lse, mB)                 # scalar

        current_iter = getattr(self, '_iter', 0)
        total_iters = getattr(self, '_max_iter', 200)
        
        # Linear annealing
        tau = tau_init + (tau_final - tau_init) * min(current_iter / total_iters, 1.0)
        
        # Compute softmax weights with annealed temperature
        w = jnp.exp(jnp.array([pA, pB]) / tau)
        w = w / (w.sum() + 1e-12)


        L_energy = weight * (w[0] * pA + w[1] * pB)
        return {
            "multi_ptme": L_energy,
            "ptme_A": pA,  # Monitor individual target energies
            "ptme_B": pB,
            "weight_A": w[0],  # Monitor softmax weights
            "weight_B": w[1],
            "tau_current": tau  # Monitor annealing schedule
        }
        

    # Đăng ký callback + trọng số
    if not any(getattr(cb, "__name__", "") == "loss_dual_ptme" for cb in self._callbacks["model"]["loss"]):
        loss_dual_ptme.__name__ = "loss_dual_ptme"
        self._callbacks["model"]["loss"].append(loss_dual_ptme)

    self.opt.setdefault("weights", {})
    self.opt["weights"]["multi_ptme"] = float(weight)
    # lưu cấu hình để dùng ở bước 2 (geodesic/overlap)
    self.opt["dual_ptme_cfg"] = {
        "chains_A": list(map(int, chains_A)),
        "chains_B": list(map(int, chains_B)),
        "binder_chain_id": int(binder_chain_id),
        "iface_thresh": float(iface_thresh),
        "tau_init": float(tau_init),
        "tau_final": float(tau_final),
    }

def add_dual_overlap_geodesic_losses(
    self,
    chains_A, chains_B, binder_chain_id,
    weight_overlap=0.2,       # λ_cap
    weight_geo=0.1,           # λ_geo
    iface_thresh=0.30,        # ngưỡng tạo S_A, S_B
    r_cut=8.0,                # bán kính lân cận CA-CA cho đồ thị bề mặt
    sigma=3.0,                # cho kernel Gaussian lân cận
    geo_min=25.0,             # d_min^geo
):
    """
    - Xây site mềm S_A, S_B từ lse (giống pTME kernel), rồi:
      * overlap Jaccard(SA, SB) → phạt trùng site
      * geodesic separation E[d_geo] giữa SA và SB → tăng khoảng cách trên bề mặt
    - Trả về 2 loss scalar: "overlap" và "geo_sep".
    """

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    def _tm_d0(N: int) -> jnp.ndarray:
        d0 = 1.24 * (jnp.maximum(jnp.asarray(N, jnp.float32), 1.0) - 15.0) ** (1.0 / 3.0) - 1.8
        return jnp.maximum(d0, jnp.asarray(1e-3, jnp.float32))

    def _bin_centers_from_breaks(breaks: jnp.ndarray | None, num_bins: int, max_err: float = 31.75) -> jnp.ndarray:
        if breaks is None:
            edges = jnp.linspace(0.0, max_err, num_bins + 1, dtype=jnp.float32)
        else:
            edges = jnp.concatenate(
                [jnp.array([0.0], jnp.float32), breaks.astype(jnp.float32), jnp.array([max_err], jnp.float32)], axis=0
            )
        return 0.5 * (edges[:-1] + edges[1:])  # (B,)

    def _tm_lse_from_pae(pae):
        logits = pae["logits"]  # (L,L,B)
        L = logits.shape[0]
        B = logits.shape[-1]
        breaks = pae.get("breaks", None)
        centers = _bin_centers_from_breaks(breaks, B)     # (B,)
        d0 = _tm_d0(L)
        g = 1.0 / (1.0 + (centers / d0) ** 2)            # (B,)
        logZ = logsumexp(logits, axis=-1)
        lse = logsumexp(logits + jnp.log(g + 1e-12)[None, None, :], axis=-1) - logZ  # (L,L)
        return lse

    def _inter_mask(chain_idx, src_list, dst_id):
        src = jnp.isin(chain_idx, jnp.array(src_list))
        dst = (chain_idx == dst_id)
        return jnp.logical_or(jnp.logical_and(src[:, None], dst[None, :]),
                              jnp.logical_and(dst[:, None], src[None, :]))  # (L,L) bool

    def _binder_mask(chain_idx, binder_id):
        return (chain_idx == binder_id)

    def _soft_iface_scores(lse, mask_bool, binder_mask):
        # lấy điểm tương tác "mềm" trên binder: score_b(i) = mean_j lse[i,j] cho các (i,j) nằm trong mask inter
        # sau đó chỉ giữ các i thuộc binder
        m = mask_bool.astype(lse.dtype)
        s_i = (lse * m).sum(axis=1) / (m.sum(axis=1) + 1e-12)   # (L,)
        s_i = jnp.where(binder_mask, s_i, 0.0)                  # chỉ binder
        # chuẩn hoá về [0,1] tương đối (không bắt buộc, nhưng giúp ổn định)
        s_min = s_i.min()
        s_max = s_i.max()
        s = (s_i - s_min) / (s_max - s_min + 1e-12)
        return s  # (L,)

    def _binary_set_from_score(s, thr):
        # S = {i | s_i >= thr}
        return (s >= thr)

    def _jaccard(a_bool, b_bool):
        a = a_bool.astype(jnp.float32)
        b = b_bool.astype(jnp.float32)
        inter = (a * b).sum()
        union = (a + b - a * b).sum()
        return inter / (union + 1e-6)

    def _pairwise_dists_CA(xyz):
        # xyz: (L, 37, 3) from structure_module; lấy CA
        ca = xyz[:, residue_constants.atom_order["CA"]]  # (L,3)
        d = jnp.sqrt(jnp.maximum(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1), 1e-12))  # (L,L)
        return d

    def _graph_resistance_distance(d_euclid, binder_mask, r_cut=8.0, sigma=3.0):
        # Đồ thị trên binder theo CA-CA; trọng số Gaussian với ngưỡng r_cut
        # Tính "resistance distance" xấp xỉ bằng pseudo-inverse Laplacian.
        bm = binder_mask.astype(bool)
        d_bb = d_euclid[bm][:, bm]     # (Lb,Lb)
        Lb = d_bb.shape[0]
        A = jnp.exp(-(d_bb ** 2) / (2 * (sigma ** 2))) * (d_bb <= r_cut)  # adjacency
        # Laplacian
        deg = jnp.diag(jnp.sum(A, axis=1))
        Lmat = deg - A + 1e-6 * jnp.eye(Lb)  # regularize
        # pseudo-inverse bằng eigh (ổn định hơn inv cho JAX)
        w, V = jnp.linalg.eigh(Lmat)
        # More robust pseudo-inverse with adaptive threshold
        w_inv = jnp.where(w > 1e-5, 1.0 / w, 0.0)  # Slightly relaxed threshold
        L_plus = (V * w_inv) @ V.T

        # Alternative: Add stronger diagonal regularization
        Lmat = deg - A + 1e-4 * jnp.eye(Lb)  # Increase from 1e-6 to 1e-4                 # (Lb,Lb)
        # resistance distance giữa i,j: r_ij = L+_ii + L+_jj - 2L+_ij
        diag = jnp.diag(L_plus)
        R = diag[:, None] + diag[None, :] - 2.0 * L_plus  # (Lb,Lb)
        return R, bm  # trả về (distance trên binder, mask vị trí binder trong toàn chuỗi)

    def _expect_geo_distance(R_binder, bm, pA, pB):
        pA_b = pA[bm]
        pB_b = pB[bm]
        
        # Check if distributions are non-empty
        sumA = pA_b.sum()
        sumB = pB_b.sum()
        
        # If either distribution is empty, return large distance (no penalty)
        valid = (sumA > 1e-8) & (sumB > 1e-8)
        
        pA_b = pA_b / jnp.maximum(sumA, 1e-12)
        pB_b = pB_b / jnp.maximum(sumB, 1e-12)
        
        E_geo = (pA_b[None, :] @ R_binder @ pB_b[:, None]).squeeze()
        
        # Return large value if invalid (no penalty applied)
        return jnp.where(valid, E_geo, 100.0)

    def loss_overlap_geo(inputs, outputs):
        pae = outputs.get("predicted_aligned_error", None)
        xyz = outputs.get("structure_module", None)
        if pae is None or ("logits" not in pae) or ("chain_index" not in inputs) or xyz is None:
            return {"overlap": jnp.asarray(0.0, jnp.float32),
                    "geo_sep": jnp.asarray(0.0, jnp.float32)}

        chain_idx = inputs["chain_index"]           # (L,)
        lse = _tm_lse_from_pae(pae)                 # (L,L)

        # masks inter cho A và B
        interA = _inter_mask(chain_idx, chains_A, binder_chain_id)
        interB = _inter_mask(chain_idx, chains_B, binder_chain_id)
        bm = _binder_mask(chain_idx, binder_chain_id)

        # score mềm trên binder
        sA = _soft_iface_scores(lse, interA, bm)    # (L,)
        sB = _soft_iface_scores(lse, interB, bm)    # (L,)

        # nhị phân hoá theo ngưỡng iface_thresh
        SA = _binary_set_from_score(sA, iface_thresh)
        SB = _binary_set_from_score(sB, iface_thresh)

        # Overlap Jaccard
        jac = _jaccard(SA, SB)                     # ∈[0,1]
        L_overlap = weight_overlap * jac

        # Geodesic: dựng đồ thị trên binder và tính resistance distance
        d_e = _pairwise_dists_CA(xyz["final_atom_positions"])  # (L,L)
        R_binder, mask_b = _graph_resistance_distance(d_e, bm, r_cut=r_cut, sigma=sigma)

        # phân bố mềm p_A, p_B (chuẩn hoá về xác suất)
        pA = jnp.maximum(sA, 0.0)
        pB = jnp.maximum(sB, 0.0)
        pA = pA / (pA.sum() + 1e-12)
        pB = pB / (pB.sum() + 1e-12)

        E_geo = _expect_geo_distance(R_binder, mask_b, pA, pB)   # kỳ vọng khoảng cách địa hình
        # loss đẩy xa: [geo_min - E_geo]_+
        L_geo = weight_geo * jax.nn.relu(geo_min - E_geo)

        return {"overlap": L_overlap, "geo_sep": L_geo}

    # đăng ký callback và weights
    loss_overlap_geo.__name__ = "loss_overlap_geo"
    self._callbacks["model"]["loss"].append(loss_overlap_geo)
    self.opt["weights"]["overlap"] = float(weight_overlap)
    self.opt["weights"]["geo_sep"] = float(weight_geo)
    # lưu cấu hình
    self.opt["dual_ptme_geom"] = {
        "iface_thresh": float(iface_thresh),
        "r_cut": float(r_cut),
        "sigma": float(sigma),
        "geo_min": float(geo_min),
    }



# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn', 
                   'multi_ptme', 'overlap', 'geo_sep']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # Close the figure
            plt.close()
