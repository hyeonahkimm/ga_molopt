oracle_array=('qed' 'drd2' 'gsk3b' 'mestranol_similarity' 'thiothixene_rediscovery' 'isomers_c9h10n2o2pf2cl' \
'celecoxib_rediscovery' 'troglitazone_rediscovery' 'albuterol_similarity' \
'zaleplon_mpo' 'valsartan_smarts' )


for oralce in "${oracle_array[@]}"
do
for seed in 1 2 3
do
# echo $oralce
CUDA_VISIBLE_DEVICES=1 python run.py ngo_gfn --oracles $oralce --seed $seed --run_name ga+rtb --config_default hparams_default.yaml --wandb online 
done
done
