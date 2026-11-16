# Example and algorithms from lecture 11

import math

def reverse_parents(parents):
    node = "Goal"
    path_rev = [node]
    while node != "Start":
        node = parents[node]
        path_rev.append(node)

    path_rev.reverse()

    return path_rev

def a_star(G, P, h):
    OPEN = { "Start": l1_dist(P["Start"], P["Goal"])}
    CLOSED = []
    parents = {}
    past_costs = {}

    for node in G: past_costs[node] = math.inf
    past_costs["Start"] = 0

    while OPEN:
        node, _ = min(OPEN.items(), key=lambda item: item[1])

        for neighbor, edge in G[node]:
            if neighbor in CLOSED: continue

            candidate_past_cost = edge + past_costs[node]

            if neighbor not in past_costs or candidate_past_cost < past_costs[neighbor]:
                past_costs[neighbor] = candidate_past_cost
                parents[neighbor] = node
                OPEN[neighbor] = past_costs[neighbor] + h(P[neighbor], P["Goal"])

        OPEN.pop(node)
        CLOSED.append(node)

        if node == "Goal":
            return (past_costs["Goal"], reverse_parents(parents))

    return math.inf # path not found

def l1_dist(a, b):
    x1, y1 = a
    x2, y2 = b

    return abs(y2 - y1) + abs(x2 - x1)

def l2_dist(a, b):
    x1, y1 = a
    x2, y2 = b

    return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

# Djikstra; this is just 
# alpha star with the 
# zero-fn as heuristic,
# which is still admissable
def djikstra(G, P):
    def h(a, b): return 0

    return a_star(G, P, h)

# grid position map for each node
P = {"A" : (0, 0), "Start": (0, 1), "C": (0, 2), "D" : (1, 1),
     "E": (2, 2), "B": (2, 1), "Goal": (2, 0)}

# graph
# for this example, we can assume 
# uniform, unit edge weights
G = {"Start": [("A", 1), ("C", 1)], "A": [("Start", 1)], "C": [("Start", 1), ("D", 1)], 
     "D": [("C", 1), ("E", 1)], "E": [("D", 1), ("B", 1)], "B":[("E", 1), ("Goal", 1)], "Goal":[("B", 1)]}

print(a_star(G, P, l1_dist))