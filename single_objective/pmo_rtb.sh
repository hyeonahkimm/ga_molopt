oracle_array=('jnk3' 'drd2' 'qed' 'gsk3b' 'celecoxib_rediscovery' 'troglitazone_rediscovery' \
        'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \
        'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \
        'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

for oralce in "${oracle_array[@]}"
do
for seed in 1 2 3
do
# echo $oralce
taskset -c 24-35 python run.py ngo --oracles $oralce --seed $seed --run_name rtb --config_default hparams_rtb.yaml --wandb online 
done
done

for oralce in "${oracle_array[@]}"
do
for seed in 1 2 3
do
# echo $oralce
taskset -c 24-35 python run.py ngo --oracles $oralce --seed $seed --run_name molga_rtb --config_default hparams_molga_rtb.yaml --wandb online 
done
done