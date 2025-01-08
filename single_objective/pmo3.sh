oracle_array=('sitagliptin_mpo' 'ranolazine_mpo' 'perindopril_mpo'  'scaffold_hop' 'deco_hop' \
'isomers_c7h8n2o2' 'median1' 'median2' 'osimertinib_mpo' 'fexofenadine_mpo' 'amlodipine_mpo')

for oralce in "${oracle_array[@]}"
do
for seed in 1 2 3
do
# echo $oralce
CUDA_VISIBLE_DEVICES=1 python run.py ngo_gfn --oracles $oralce --seed $seed --run_name ga+rtb --config_default hparams_default.yaml --wandb online 
done
done
