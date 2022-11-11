import subprocess
import time

import gpu_check

curriculum_agent_reward_threshs = [
    .05, .1, .15, .2, .25
]
curriculum_agent_success_rates = [
    .5, .55, .6, .65
]
gpus_to_use = [0, 1, 3, 4, 5]

scheduling_thresh = 4000

for curriculum_agent_reward_thresh in curriculum_agent_reward_threshs:
    for curriculum_agent_success_rate in curriculum_agent_success_rates:
        while True:
            gpus = gpu_check.getGPUs()
            found = False
            for gpu_idx, gpu in enumerate(gpus):
                if gpu_idx not in gpus_to_use:
                    continue

                if gpu.memoryFree > scheduling_thresh:
                    bashCommand = f"CUDA_VISIBLE_DEVICES={gpu_idx} PYTHONPATH=/home/yppatel/misc/clean_idp_rl/src/ python examples/chignolin_example.py --curriculum_agent_reward_thresh {curriculum_agent_reward_thresh} --curriculum_agent_success_rate {curriculum_agent_success_rate}"
                    print(bashCommand)
                    
                    # subprocess.run(['bash','-c', bashCommand])
                    subprocess.Popen([bashCommand], shell=True, stdin=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(30) # wait for GPU scheduling to occur (compute normalizers)
                    found = True
                    break
            
            if not found:
                time.sleep(10)
            else:
                break