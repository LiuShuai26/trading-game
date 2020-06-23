import sys
import subprocess
import time

root_path = '/home/shuai/trading-game/spinningup/data/'

exp_name = "ppo-trainning_set=54-model=mlp-obs_dim=26-as15-auto_follow=0-burn_in-3000-fs=1-ts=1-ss=1.5-ap=0.4dl=30clip=5-lr=4e-05"

args = [sys.executable] + [root_path+'subptest.py', '--exp_name', exp_name]

tests = [subprocess.Popen(args+['--start', str(start)], stdout=subprocess.PIPE) for start in range(1, 4+1)]
output = [int(str(test.communicate()[0]).split('\\n')[-2]) for test in tests]
# # scores = [test.returncode for test in tests]
# # print(scores)
print(output)
average_score = sum(output)/len(output)
print(average_score)

# tests = [subprocess.Popen(args+['--start', str(start)]) for start in range(1, 4+1)]
# [test.wait() for test in tests]
if average_score < 150:
    subprocess.run(["mv", root_path+exp_name + '/' + exp_name + '_s0/last_tf1_save', root_path+exp_name + '/' + exp_name + '_s0/tf1_save' + str(average_score)])
