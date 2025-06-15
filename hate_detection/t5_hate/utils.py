import json

def load_train_data(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def load_test_data(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def convert_train_to_t5_format(data, task_prefix="仇恨识别: "):
    """
    把训练数据转换成T5需要的输入格式
    例：输入：{"content": "...", "output": "..."}，
        输出： task_prefix + content 和 output 两部分
    """
    inputs = []
    outputs = []
    for item in data:
        text = item["content"]
        label = item.get("output", "")  # 有些测试集可能没有output字段
        inputs.append(task_prefix + text)
        outputs.append(label)
    return inputs, outputs
