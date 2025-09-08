from kubernetes import client, config, watch

config.load_kube_config()
objs_api = client.CustomObjectsApi()
core_api = client.CoreV1Api()

nodes = core_api.list_node().items
print(nodes)

def is_shutting_down(node):
    for c in node.status.conditions or []:
        if c.type == "Ready" and c.status != "True" and c.reason == "Shutdown":
            return True
    return False

total_everything = len(nodes)
active_nodes = [n for n in nodes if not is_shutting_down(n)]
total_nodes = len(active_nodes)
ready_nodes = sum(
    1 for n in active_nodes
    if any(c.type == "Ready" and c.status == "True" for c in n.status.conditions)
)