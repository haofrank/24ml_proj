# Connect and edit in VS Code

## Step 1: generate you ssh key local, ensure the id_rsa.pub exists

## Step 2: `ssh greene`, then `ssh burst`

## Step 3: copy your public ssh key to ~/.ssh/authorized_keys, e.g.
```
# on your local pc
cat .ssh/id_rsa.pub

# then go to burst terminal
vim .ssh/authorized_keys

# copy public key into that file and save
```


## Step 4: apply for nodes using sbatch, and get the node number
```
sbatch --account=csci_ga_3033_077-2024sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash # submit application

squeue --me # get node number

# wait until the status of the node (ST) change to "R"
# example output:
[NetID@log-burst ~]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            167330 n1s8-v100     wrap   hj2533  R       0:41      1 b-3-17

```

## Step 5: edit your local ~/.ssh/config, add or modify:
```
Host greene
    HostName greene.hpc.nyu.edu
    User NetID
    IdentityFile C:\users\liyihan\.ssh\id_rsa # change to your path to id_rsa (private key file)
Host burst
    HostName log-burst.hpc.nyu.edu
    User NetID
    IdentityFile C:\users\liyihan\.ssh\id_rsa

Host awesome-node
    HostName b-3-17  # this should be the node number you got in the previous step, need to modify every time
    User NetID
    ProxyJump burst
```

## Step 6: try ssh connection in VS Code

## Checklist
- Can you ssh greene/burst?
  - If not, check VPN if you are not on campus
 
- Check local ~/.ssh/known_hosts
  - If you Ctrl+F find the node number, delete that line and retry
 

 


