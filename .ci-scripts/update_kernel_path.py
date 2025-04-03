import jupyter_client
import json
import os

specs = jupyter_client.kernelspec.find_kernel_specs()
kernel_dir = specs['quandary-env']
kernel_json_path = os.path.join(kernel_dir, "kernel.json")

with open(kernel_json_path, 'r') as f:
    kernel_spec = json.load(f)

if 'env' not in kernel_spec:
    kernel_spec['env'] = {}

kernel_spec['env']['PATH'] = "/root/bin:" + os.environ.get('PATH', '')

with open(kernel_json_path, 'w') as f:
    json.dump(kernel_spec, f, indent=2)

print(f"Updated kernel.json at {kernel_json_path} with PATH prepended")
