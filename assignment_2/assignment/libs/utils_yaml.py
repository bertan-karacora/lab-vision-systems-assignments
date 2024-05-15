import yaml


def read(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    return config


class DumperIndent(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(DumperIndent, self).increase_indent(flow, False)


def save(attributes, path):
    with open(path, "w+") as file:
        yaml.dump(attributes, file, Dumper=DumperIndent, default_flow_style=False, indent=4)
