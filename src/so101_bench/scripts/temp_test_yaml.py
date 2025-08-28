import yaml

data = yaml.safe_load(
    open("/home/melon/sherry/so101_bench/datasets/tasks/pick_and_place_block.yaml")
)

print(data)
