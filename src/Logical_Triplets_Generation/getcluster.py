from collections import defaultdict,deque


def cluster_entities(triples):
    adj_list = defaultdict(set)
    entities = set()
    
    for head, relation, tail in triples:
        adj_list[head].add(relation+'head')
        adj_list[tail].add(relation+'tail')
        entities.update([head, tail])

    clusters_dict = defaultdict(list)
    for entity in adj_list:
        key = frozenset(adj_list[entity]) 
        clusters_dict[key].append(entity)
    
    clusters = list(clusters_dict.values())
    
    return clusters

# Function to build tree for a given entity
def build_tree(entity, triples):
    # Build a graph where each entity points to its connected entities
    graph = defaultdict(list)
    
    for subject, predicate, obj in triples:
        graph[subject].append((predicate, obj))
        graph[obj].append((predicate, subject))  # since it's undirected
    
    # Function to perform DFS and construct the tree
    def dfs(current, path, visited):
        # Store the full path from the root (entity) to the current entity
        result = []
        visited.add(current)  # Mark the current entity as visited
        
        for predicate, connected_entity in graph[current]:
            # Avoid cycles by skipping entities already in the path
            if connected_entity not in visited:
                new_path = path + [(current, predicate, connected_entity)]
                result.append(new_path)
                result.extend(dfs(connected_entity, new_path, visited))
        
        visited.remove(current)  # Unmark the current entity to allow exploration from other paths
        return result

    # Start DFS from the root entity
    visited = set()  # Keep track of visited nodes to avoid cycles
    return dfs(entity, [], visited)



# Process clustering logic
def split_clusters_by_leaves(clusters, entity2leaves, entity2tree):
    new_clusters = []
    cluster2tree = {}
    for cluster in clusters:
        # Get leaves for each entity in the current cluster
        leaves_set = [entity2leaves[entity] for entity in cluster if entity in entity2leaves]
        
        # If there are inconsistent leaves in the cluster, further split it
        if len(set(frozenset(leaves) for leaves in leaves_set)) > 1:
            # Split into smaller clusters based on leaves
            leaves_to_group = {}
            for entity, leaves in zip(cluster, leaves_set):
                if frozenset(leaves) not in leaves_to_group:
                    leaves_to_group[frozenset(leaves)] = []
                leaves_to_group[frozenset(leaves)].append(entity)
                cluster2tree[entity] = entity2tree[entity]

            # Add split smaller clusters to new cluster collection
            new_clusters.extend(leaves_to_group.values())
        else:
            new_clusters.append(cluster)
    
    return new_clusters


def getCluster(s):
    triples = s
    clusters = cluster_entities(s)
    entity2cluster = {}
    for cluster in clusters:
        for entity in cluster:
            entity2cluster[entity] = '#'.join(set(cluster))

    cluster_triples = []
    for item in triples:
        cluster_item = [entity2cluster[item[0]],item[1],entity2cluster[item[2]]]
        if cluster_item not in cluster_triples:
            cluster_triples.append(cluster_item)

    entity_cluster_check_map = {}
    for entity in set(entity2cluster.values()):
        tmp_tree = build_tree(entity, cluster_triples)
        tree = []
        for num,item in enumerate(tmp_tree):
            if num == len(tmp_tree)-1:
                tree.append(item)
                break
            next_item = tmp_tree[num+1]
            if set(item).issubset(set(next_item)):
                continue
            else:
                tree.append(item)
        cluster_relation_paths = [[j[1] for j in i] for i in tree]
        for e in entity.split('#'):
            entity_cluster_check_map[e] = cluster_relation_paths

    # Loop through clusters and build tree for each element
    entity2leaves = {}
    entity2tree = {}
    entity2triples = {}
    for cluster in clusters:
        if len(cluster) <=1 :
            continue
        for entity in cluster:
            other_entity = [i for i in cluster if i!=entity]
            tmp_triples = [i for i in triples if i[0] not in other_entity and i[1] not in other_entity]
            entity2triples[entity] = tmp_triples
            tmp_tree = build_tree(entity, tmp_triples)
            tree = []
            for branch in tmp_tree:
                barnch_relation_path = [i[1] for i in branch]
                if barnch_relation_path in entity_cluster_check_map[entity]:
                    tree.append(branch)
            
            leaves = set([i[-1][-1] for i in tree])
            entity2leaves[entity] = leaves
            entity2tree[entity] = tree

    split_clusters = split_clusters_by_leaves(clusters, entity2leaves,entity2tree)

    return split_clusters,entity2triples


def calculate_group_relationships(triplets, groups):
    entity_d = {}
    entity_to_group = {}
    for idx, group in enumerate(groups):
        for entity in group:
            entity_to_group[entity] = idx

    group_relationships = defaultdict(int)

    for head, relation, tail in triplets:
        if head not in entity_d.keys():
            entity_d[head] = 0
            entity_d[head] += 1
        else: entity_d[head] += 1
        if tail not in entity_d.keys():
            entity_d[tail] = 0
            entity_d[tail] += 1
        else: entity_d[tail] += 1

        if head in entity_to_group and tail in entity_to_group:
            head_group = entity_to_group[head]
            tail_group = entity_to_group[tail]
            group_relationships[(head_group, relation, tail_group)] += 1

    return group_relationships,entity_d


def find_paths_to_leaf(group, group_relationships, leaf_list):
    paths = []
    leaf = [i for i in leaf_list if i != group]
    visited = set()

    def dfs(current_group, path, leaf_list=leaf):
        if current_group in leaf_list:
            paths.append(path[:])
            return

        visited.add(current_group)

        for (head_group, relation, tail_group), count in group_relationships.items():
            if (head_group == current_group or tail_group == current_group) and (head_group, relation, tail_group) not in path:
                next_group = tail_group if head_group == current_group else head_group
                if next_group not in visited:
                    path.append((head_group, relation, tail_group))
                    dfs(next_group, path)
                    path.pop()

        visited.remove(current_group)

    dfs(group, [])
    return paths

def find_paths_to_leaf(group, group_relationships, leaf_list):
    paths = []
    leaf = [i for i in leaf_list if i != group]
    leaf_set = set(leaf)
    
    queue = deque([(group, [])])
    
    while queue:
        current_group, path = queue.popleft()
        visited = {node for edge in path for node in (edge[0], edge[2])}
        
        has_next = False
        for (head_group, relation, tail_group), count in group_relationships.items():
            if head_group == current_group or tail_group == current_group:
                next_group = tail_group if head_group == current_group else head_group
                if next_group not in visited:
                    has_next = True
                    new_path = path + [(head_group, relation, tail_group)]
                    queue.append((next_group, new_path))
        
        if not has_next and current_group in leaf_set:
            paths.append(path)
    
    return paths



def determine_q_list(groups, leaf_list, group_relationships,triplets,tailrel2head):
    q_list = []

    for a in range(len(groups)):
        paths = find_paths_to_leaf(a, group_relationships, leaf_list)
        can_add_to_q_list = True

        for path in paths:
            if len(path) > 1:
                for idx, (head_group, relation, tail_group) in enumerate(path):

                    if group_relationships[(head_group, relation, tail_group)] > 1:
                        break

                    if idx == len(path) - 1:
                        (head_group, relation, tail_group) = path[idx]
                        if tail_group in leaf_list:
                            if not findLeafTriplets(head_group,tail_group,triplets,groups,tailrel2head):
                                can_add_to_q_list = False

                        if head_group in leaf_list:
                            if group_relationships[(head_group, relation, tail_group)]==1:
                                can_add_to_q_list = False

            else:
                (head_group, relation, tail_group) = path[0]

                if a == tail_group:
                    if group_relationships[(head_group, relation, tail_group)] == 1:
                        can_add_to_q_list = False

                elif a == head_group:
                    if group_relationships[(head_group, relation, tail_group)] == 1:
                        if not findLeafTriplets(head_group,tail_group,triplets,groups,tailrel2head):
                            can_add_to_q_list = False

        if can_add_to_q_list:
            q_list.append(a)

    return q_list


def findLeafTriplets(a,tail_group,triplets,groups,tailrel2head):
    count = 0
    for head, relation, tail in triplets:
        if head in groups[a] and tail in groups[tail_group]:
            r = relation
            tail_find = tail
    count = len(tailrel2head[(tail_find,r)])
    if count>1:
        return True
    else:
        return False

triples = []

with open("../Subgraph_Selection/initial_source/FB15K.txt", "r") as file:
    for line in file:
        line = line.strip()
        data_tuple = tuple(item.strip() for item in line.split(","))
        if data_tuple[0] != data_tuple[2]:
            triples.append(data_tuple)

node2rel_in = {}
node2rel_out = {}
rel2triple = {}
headrel2tail = {}
tailrel2head = {}


for (head,r,tail) in triples:
    if head not in node2rel_out.keys():
        node2rel_out[head] = []
        node2rel_out[head].append(r)
    else:
        node2rel_out[head].append(r)

    if tail not in node2rel_in.keys():
        node2rel_in[tail] = []
        node2rel_in[tail].append(r)
    else:
        node2rel_in[tail].append(r)

    if r not in rel2triple.keys():
        rel2triple[r] = []
        rel2triple[r].append([head,tail])
    else:
        rel2triple[r].append([head,tail])

    if (head,r) not in headrel2tail.keys():
        headrel2tail[(head,r)] = []
        headrel2tail[(head,r)].append(tail)
    else:
        headrel2tail[(head,r)].append(tail)

    if (tail,r) not in tailrel2head.keys():
        tailrel2head[(tail,r)] = []
        tailrel2head[(tail,r)].append(head)
    else:
        tailrel2head[(tail,r)].append(head)


def getEntityCluster(triplets,tailrel2head):

    groups,cluster2tree = getCluster(triplets)

    group_relationships,entity_d = calculate_group_relationships(triplets, groups)


    leaf_list = []
    for idx,item in enumerate(groups):
        flag = 0
        for e in item:
            if entity_d[e] > 1:
                flag = 1
                break
        if flag == 0:
            leaf_list.append(idx)

    q_list = determine_q_list(groups, leaf_list, group_relationships,triplets,tailrel2head)

    QCluster = [groups[q] for q in q_list]

    return QCluster,groups,cluster2tree

