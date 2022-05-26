# the Implementation of Decentralized Federated Learning under Selective Partial Sharing

## the environment
The needed environment is in requirenments.txt

## file description
1. main_odr.py; the main function;
2. utils; the auxiliary function (including sampling, loading data, model selection, etc.);
3. models; the folder containing the training code; (1. Nets.py defines the used all model strucuture; 2. OdrUpdate.py is the pipeline of the training process; 3.test.py is the code for evaluation)

## selected parameters description
1. num_users; the number of users
2. shard_per_user; the classes of each client (when sampling data for all clients)
3. bs: bacth_size for training
4. model; the used model strucure;
5. dataset; the used dataset;
6. eps; the scale of distribution shift
7. alpha_odr and beta_odr; the weight of $\alpha$ and $\beta$
8. odr whether useing ODR term

## conduct the experiments
sh run.sh